"""House Price Intelligence: analysis, visualization and price estimation.
Run: streamlit run house_price_project.py
Keep house_prices.csv in the same directory (or use the file uploader).
"""
import re, os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Intelligence", page_icon="🏠", layout="wide")
st.title("🏠 House Price Intelligence")
st.caption("Explore Indian residential property listings, understand market patterns, and estimate indicative prices.")

@st.cache_data(show_spinner=False)
def load_data(file_bytes=None):
    if file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    local='house_prices.csv'
    if os.path.exists(local): return pd.read_csv(local, low_memory=False)
    return None

def parse_price(v):
    if pd.isna(v): return np.nan
    s=str(v).lower().replace(',','').replace('₹','').replace('rs.','').replace('inr','').strip()
    m=re.search(r'([-+]?[0-9]*\.?[0-9]+)\s*(cr|crore|crores|lac|lakh|lakhs|k)?',s)
    if not m: return np.nan
    n=float(m.group(1)); unit=m.group(2)
    if unit in ('cr','crore','crores'): n*=10_000_000
    elif unit in ('lac','lakh','lakhs'): n*=100_000
    elif unit=='k': n*=1000
    return n

def parse_area(v):
    if pd.isna(v): return np.nan
    s=str(v).lower().replace(',','').strip()
    m=re.search(r'([0-9]*\.?[0-9]+)',s)
    if not m:return np.nan
    n=float(m.group(1))
    if 'sqyrd' in s or 'sq yd' in s or 'square yard' in s: n*=9
    elif 'sqm' in s or 'sq m' in s or 'square meter' in s: n*=10.7639
    elif 'acre' in s: n*=43560
    return n

def extract_bhk(v):
    if pd.isna(v): return np.nan
    m=re.search(r'(\d+(?:\.\d+)?)\s*(?:bhk|bedroom)',str(v).lower())
    return float(m.group(1)) if m else np.nan

def floor_parts(v):
    if pd.isna(v): return (np.nan,np.nan)
    s=str(v).lower(); m=re.search(r'(\d+)\s*out of\s*(\d+)',s)
    if m:return float(m.group(1)),float(m.group(2))
    if 'ground' in s:return 0,np.nan
    m=re.search(r'^\s*(\d+)',s)
    return (float(m.group(1)),np.nan) if m else (np.nan,np.nan)

def prepare(raw):
    d=raw.copy()
    required=['Amount(in rupees)','location','Title']
    missing=[c for c in required if c not in d.columns]
    if missing: raise ValueError('Dataset is missing required columns: '+', '.join(missing))
    d['target_price_inr']=d['Amount(in rupees)'].map(parse_price)
    d['carpet_area_sqft']=d['Carpet Area'].map(parse_area) if 'Carpet Area' in d else np.nan
    d['super_area_sqft']=d['Super Area'].map(parse_area) if 'Super Area' in d else np.nan
    d['bhk']=d['Title'].map(extract_bhk)
    floors=d['Floor'].map(floor_parts) if 'Floor' in d else [(np.nan,np.nan)]*len(d)
    d['floor_number']=[x[0] for x in floors]; d['total_floors']=[x[1] for x in floors]
    d['balcony_count']=pd.to_numeric(d['Balcony'].astype(str).str.extract(r'(\d+)')[0],errors='coerce') if 'Balcony' in d else np.nan
    d['parking_count']=pd.to_numeric(d['Car Parking'].astype(str).str.extract(r'(\d+)')[0],errors='coerce') if 'Car Parking' in d else np.nan
    d['bathroom_count']=pd.to_numeric(d['Bathroom'],errors='coerce') if 'Bathroom' in d else np.nan
    return d

uploaded=st.sidebar.file_uploader('Upload house_prices.csv',type=['csv'])
try:
    raw=load_data(uploaded.getvalue() if uploaded else None)
except Exception as e:
    st.error(f'Could not read the dataset: {e}'); st.stop()
if raw is None:
    st.info('Place `house_prices.csv` next to this Python file, or upload it using the sidebar.')
    st.stop()
with st.spinner('Preparing the dataset…'):
    data=prepare(raw)
valid=data[data['target_price_inr'].between(100_000,500_000_000)].copy()
if valid.empty:
    st.error('No valid prices were parsed. Check the Amount(in rupees) column format.'); st.stop()

with st.sidebar:
    st.header('Filters')
    locations=sorted(valid['location'].dropna().astype(str).unique())
    chosen=st.multiselect('Location',locations,default=locations[:min(8,len(locations))])
    if not chosen: chosen=locations
    min_lakh=float(valid['target_price_inr'].quantile(.01)/100000)
    max_lakh=float(valid['target_price_inr'].quantile(.99)/100000)
    price_range=st.slider('Listed price range (₹ lakh)',min_value=max(1,int(min_lakh)),max_value=max(int(max_lakh),int(min_lakh)+1),value=(max(1,int(min_lakh)),max(int(max_lakh),int(min_lakh)+1)))
filtered=valid[valid['location'].astype(str).isin(chosen)]
filtered=filtered[filtered['target_price_inr'].between(price_range[0]*100000,price_range[1]*100000)]
if filtered.empty: filtered=valid[valid['location'].astype(str).isin(chosen)]

c1,c2,c3,c4=st.columns(4)
c1.metric('Listings in view',f'{len(filtered):,}')
c2.metric('Median listed price',f'₹{filtered.target_price_inr.median()/100000:.1f} lakh')
c3.metric('Median price / sq ft',f'₹{filtered["Price (in rupees)"].replace([np.inf,-np.inf],np.nan).dropna().median():,.0f}' if 'Price (in rupees)' in filtered and filtered['Price (in rupees)'].notna().any() else 'N/A')
c4.metric('Locations represented',f'{filtered.location.nunique():,}')

tab1,tab2,tab3,tab4=st.tabs(['Executive Overview','Market Analysis','Data Quality','Price Estimator'])
with tab1:
    left,right=st.columns(2)
    with left:
        st.subheader('Listed-price distribution')
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(filtered.target_price_inr/100000,bins=45); ax.set_xlabel('Price (₹ lakh)'); ax.set_ylabel('Listings'); ax.set_title('Distribution of listed prices'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader('Median price by location')
        by=filtered.groupby('location').target_price_inr.agg(['median','count']).query('count >= 20').sort_values('median',ascending=False).head(12).sort_values('median')
        if not by.empty:
            fig,ax=plt.subplots(figsize=(7,4)); ax.barh(by.index,by['median']/100000); ax.set_xlabel('Median listed price (₹ lakh)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader('Key findings from the current selection')
    most=filtered.location.value_counts().head(1)
    if len(most): st.write(f"- **Largest listing count:** {most.index[0]} ({int(most.iloc[0]):,} listings in the current view).")
    st.write(f"- **Price spread:** 25th percentile ₹{filtered.target_price_inr.quantile(.25)/100000:.1f} lakh; 75th percentile ₹{filtered.target_price_inr.quantile(.75)/100000:.1f} lakh.")
    st.write(f"- **Area coverage:** {filtered[['carpet_area_sqft','super_area_sqft']].notna().any(axis=1).mean()*100:.1f}% of listings have at least one parsed area measure.")
    st.caption('These are listing-price patterns, not verified transaction prices or financial advice. Differences between locations may reflect property mix and data quality.')
with tab2:
    st.subheader('Compare locations')
    summary=filtered.groupby('location').agg(Listings=('target_price_inr','size'),Median_price_INR=('target_price_inr','median'),Mean_price_INR=('target_price_inr','mean'),Median_price_per_sqft=('Price (in rupees)','median')).sort_values('Median_price_INR',ascending=False)
    st.dataframe(summary.style.format({'Median_price_INR':'₹{:,.0f}','Mean_price_INR':'₹{:,.0f}','Median_price_per_sqft':'₹{:,.0f}'}),use_container_width=True)
    st.subheader('Property attributes')
    attribute=st.selectbox('Break down by', [c for c in ['Status','Transaction','Furnishing','Bathroom','Balcony','Ownership'] if c in filtered.columns])
    st.dataframe(filtered.groupby(attribute,dropna=False).agg(Listings=('target_price_inr','size'),Median_price_INR=('target_price_inr','median')).sort_values('Listings',ascending=False).style.format({'Median_price_INR':'₹{:,.0f}'}),use_container_width=True)
with tab3:
    st.subheader('Missing data by column')
    miss=(raw.isna().mean()*100).sort_values(ascending=False).head(15)
    fig,ax=plt.subplots(figsize=(8,5)); miss.sort_values().plot(kind='barh',ax=ax); ax.set_xlabel('Missing values (%)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Raw dataset shape:',raw.shape)
    st.write('Duplicate rows:',int(raw.duplicated().sum()))
    st.write('Unparsed or missing listed prices:',int(data.target_price_inr.isna().sum()))
    st.write('Rows outside the broad price validity range:',int((~data.target_price_inr.between(100000,500000000)).sum()))
    st.caption('Missing values are not automatically evidence of bad data. The prediction pipeline imputes selected features and excludes rows without a usable target.')
with tab4:
    st.subheader('Train a model on this dataset')
    st.warning('The result is an indicative estimate from historical listings, not an appraisal. Do not use it as the sole basis for a purchase or investment.')
    max_train=min(18000,len(valid))
    train_n=st.slider('Maximum rows used for model training',min_value=min(1000,max_train),max_value=max_train,value=min(10000,max_train),step=1000 if max_train>=2000 else 100)
    num_features=['carpet_area_sqft','super_area_sqft','bhk','bathroom_count','balcony_count','floor_number','total_floors','parking_count']
    cat_features=['location','Status','Transaction','Furnishing','facing','Ownership']
    model_rows=valid[valid.target_price_inr.between(500000,200000000)].copy()
    model_rows=model_rows[model_rows[['carpet_area_sqft','super_area_sqft']].notna().any(axis=1)]
    if len(model_rows)>train_n: model_rows=model_rows.sample(train_n,random_state=42)
    if len(model_rows)<100:
        st.error('Not enough usable rows with a parsed price and area to train the model.')
    else:
        if st.button('Train / evaluate model',type='primary'):
            with st.spinner('Training and evaluating the model…'):
                X=model_rows[num_features+cat_features].copy(); y=model_rows.target_price_inr.astype(float)
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
                pre=ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),num_features),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=25,max_categories=30))]),cat_features)])
                pipe=Pipeline([('preprocess',pre),('model',RandomForestRegressor(n_estimators=100,max_depth=22,min_samples_leaf=3,n_jobs=-1,random_state=42))])
                pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
                a,b,c=st.columns(3); a.metric('Mean absolute error',f'₹{mean_absolute_error(yte,pred):,.0f}'); b.metric('Root mean squared error',f'₹{np.sqrt(mean_squared_error(yte,pred)):,.0f}'); c.metric('R² score',f'{r2_score(yte,pred):.3f}')
                st.caption(f'Test set: {len(Xte):,} rows. Metrics can change with the dataset and sample. Compare against a simple baseline before making strong claims.')
                st.session_state['trained_model']=pipe
    st.markdown('#### Estimate a property price')
    st.caption('Train the model above first. Unknown details can be left blank where supported; estimates are less reliable when key inputs are missing.')
    with st.form('predict_form'):
        loc=st.selectbox('Location',sorted(valid.location.dropna().astype(str).unique()))
        area=st.number_input('Carpet area (sq ft)',min_value=0.0,value=800.0,step=50.0)
        super_area=st.number_input('Super area (sq ft, optional)',min_value=0.0,value=0.0,step=50.0)
        bhk=st.selectbox('Bedrooms (BHK)',[1,2,3,4,5,6])
        baths=st.selectbox('Bathrooms',[1,2,3,4,5,6])
        status=st.selectbox('Status',['Ready to Move','Under Construction'])
        transaction=st.selectbox('Transaction',['Resale','New Property','Other'])
        furnishing=st.selectbox('Furnishing',['Unfurnished','Semi-Furnished','Furnished','Unknown'])
        facing=st.selectbox('Facing',['Unknown','East','West','North','South','North-East','North-West','South-East','South-West'])
        ownership=st.selectbox('Ownership',['Unknown','Freehold','Leasehold','Co-operative Society','Power of Attorney'])
        submitted=st.form_submit_button('Estimate price')
    if submitted:
        pipe=st.session_state.get('trained_model')
        if pipe is None: st.error('Train/evaluate the model first.')
        else:
            row=pd.DataFrame([{'carpet_area_sqft':area if area>0 else np.nan,'super_area_sqft':super_area if super_area>0 else np.nan,'bhk':float(bhk),'bathroom_count':float(baths),'balcony_count':np.nan,'floor_number':np.nan,'total_floors':np.nan,'parking_count':np.nan,'location':loc,'Status':status,'Transaction':transaction,'Furnishing':furnishing if furnishing!='Unknown' else np.nan,'facing':facing if facing!='Unknown' else np.nan,'Ownership':ownership if ownership!='Unknown' else np.nan}])
            estimate=float(pipe.predict(row)[0])
            st.success(f'Indicative estimated listing price: ₹{estimate/100000:.1f} lakh (₹{estimate:,.0f})')
            st.caption('Model output depends on historical listing data and can be inaccurate for unusual properties or changing market conditions.')

st.divider()
st.caption('Project note: prices are parsed from the `Amount(in rupees)` listing field. `Price (in rupees)` is treated as a separate price-per-unit indicator and is not used as a prediction feature to reduce target leakage.')
