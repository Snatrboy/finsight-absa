import os,json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from engine import predict,normalize_model_v2
D=Path(__file__).parent/'data'
st.set_page_config(page_title='FinSight | ABSA',page_icon='📊',layout='wide')
@st.cache_data
def csv(name): return pd.read_csv(D/name)
registry=json.loads((D/'registry.json').read_text()); connection=json.loads((D/'dashboard_connection.json').read_text())
metrics=csv('deployment_metrics.csv'); aspects=registry[0]['aspects']; labels=registry[0]['labels']; polarity=['Tidak dibahas','Negatif','Netral','Positif']; colors=['#94a3b8','#ef4444','#eab308','#14b8a6']
with st.sidebar:
 st.title('FinSight / ABSA'); st.caption('FINANCIAL INTELLIGENCE · RESEARCH')
 page=st.radio('Navigasi',['Overview','Live Prediction','Batch Prediction','Model Performance','Dataset Explorer','Methodology'])
 dark=st.toggle('Dark mode'); st.divider(); st.caption('36 konfigurasi • 6 aspek • Seed 42')
bg,fg,panel=('#0b1220','#e2e8f0','#162235') if dark else ('#f4f7fb','#14243b','#ffffff')
st.markdown(f'''<style>.stApp{{background:{bg};color:{fg}}} [data-testid="stSidebar"], [data-testid="stMetric"]{{background:{panel};color:{fg}}} [data-testid="stMetric"]{{padding:18px;border:1px solid #94a3b844;border-radius:14px}} h1,h2,h3,p,label,[data-testid="stMetricValue"],[data-testid="stMetricLabel"]{{color:{fg}!important}} .block-container{{padding-top:2rem;max-width:1450px}} </style>''',unsafe_allow_html=True)
def chart(fig):
 fig.update_layout(template='plotly_dark' if dark else 'plotly_white',paper_bgcolor=panel,plot_bgcolor=panel,font_color=fg,margin=dict(l=20,r=20,t=30,b=20)); st.plotly_chart(fig,use_container_width=True)
def heading(title,desc): st.title(title); st.caption(desc); st.divider()
def choose():
 phase=st.radio('Fase eksperimen',['A','B'],horizontal=True,index=1); opts=[m for m in registry if m['phase']==phase]; default=next((i for i,m in enumerate(opts) if m['family']=='lr' and m['mode']=='tanpa'),0)
 return st.selectbox('Konfigurasi model',opts,index=default,format_func=lambda m:m['name']+' — '+m['mode'])
def run(m,texts):
 token=os.getenv('HF_TOKEN');
 if not token:
  try: token=st.secrets.get('HF_TOKEN')
  except Exception: pass
 if not token and not os.getenv('FINSIGHT_MODEL_DIR'): raise ValueError('Atur HF_TOKEN pada Secrets atau FINSIGHT_MODEL_DIR untuk model lokal.')
 return predict(m,texts,connection,token)
def dist():
 d=csv('distribusi.csv').copy(); d['Sentimen']=d.kelas.map(dict(enumerate(polarity))); d['Aspek']=d.aspek.str[2:].map(aspects); return d
if page=='Overview':
 heading('Ringkasan penelitian','Analisis sentimen berbasis aspek pada ulasan aplikasi keuangan')
 for c,(a,b) in zip(st.columns(4),[('Ulasan mentah','100.000'),('Ulasan bersih','68.549'),('Gold label','1.060'),('Konfigurasi','36')]): c.metric(a,b)
 a,b=st.columns(2)
 with a:
  st.subheader('Ulasan per aplikasi'); chart(px.bar(csv('ulasan_per_aplikasi.csv'),x='ulasan_bersih',y='app',orientation='h',color_discrete_sequence=['#0d9488']))
 with b:
  st.subheader('Distribusi label seluruh data'); d=dist().groupby('Sentimen',as_index=False).jumlah.sum(); chart(px.pie(d,names='Sentimen',values='jumlah',hole=.65,color='Sentimen',color_discrete_map=dict(zip(polarity,colors))))
 st.info('Distribusi seluruh data mencakup label model, bukan seluruhnya anotasi manusia.')
 st.subheader('Performa deployment terbaik (OOD)'); st.dataframe(metrics.query("split=='test_ood'").nlargest(5,'F1_pasangan')[['model','phase','mode','F1_pasangan']],hide_index=True,use_container_width=True)
elif page=='Live Prediction':
 heading('Prediksi langsung','Uji ulasan baru dengan model terlatih. Probabilitas bukan jaminan kebenaran.')
 m=choose(); text=st.text_area('Ulasan',placeholder='Saldo saya hilang dan transfer gagal',max_chars=5000)
 if st.button('Analisis ulasan',type='primary'):
  if not normalize_model_v2(text): st.warning('Masukkan ulasan yang valid.')
  else:
   try:
    with st.spinner('Memuat model dan melakukan inferensi...'): prob=run(m,[text])[0]
    st.caption('Teks normalisasi: '+normalize_model_v2(text)); rows=[]
    for j,a in enumerate(aspects.values()): rows.append({'Aspek':a,'Sentimen':polarity[prob[j].argmax()],'Probabilitas':float(prob[j].max())})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,column_config={'Probabilitas':st.column_config.ProgressColumn(format='%.2f',min_value=0,max_value=1)})
    chart(px.bar(pd.DataFrame(prob,index=list(aspects.values()),columns=polarity).reset_index().melt(id_vars='index',var_name='Sentimen',value_name='Probabilitas'),x='Probabilitas',y='index',color='Sentimen',orientation='h',color_discrete_map=dict(zip(polarity,colors))))
   except Exception as e: st.error('Inferensi gagal: '+type(e).__name__+'. Periksa token, koneksi, dependensi, atau RAM.')
elif page=='Batch Prediction':
 heading('Prediksi batch','Unggah CSV UTF-8, maksimal 500 ulasan per proses untuk hosting gratis.')
 m=choose(); f=st.file_uploader('File CSV',type=['csv']); st.download_button('Contoh CSV','ulasan\nTransfer gagal\nLogin mudah\n','contoh.csv')
 if f:
  try:
   if f.size>5_000_000: raise ValueError('File maksimal 5 MB')
   df=pd.read_csv(f); col=st.selectbox('Kolom ulasan',df.columns); st.dataframe(df.head(),hide_index=True)
   if st.button('Proses batch',type='primary'):
    texts=df[col].fillna('').astype(str).tolist()
    if not texts or len(texts)>500 or any(len(t)>5000 or not normalize_model_v2(t) for t in texts): raise ValueError('Gunakan 1–500 ulasan valid, maksimal 5000 karakter per ulasan.')
    bar=st.progress(0); chunks=[]
    for i in range(0,len(texts),8): chunks.append(run(m,texts[i:i+8])); bar.progress(min(1,(i+8)/len(texts)))
    probs=np.concatenate(chunks); out=df.copy()
    for j,k in enumerate(labels): out['pred_'+k]=[polarity[x] for x in probs[:,j].argmax(-1)]; out['prob_'+k]=probs[:,j].max(-1)
    st.session_state['batch_result']=out.to_csv(index=False).encode('utf-8-sig'); st.dataframe(out.head(30),hide_index=True)
  except Exception as e: st.error(str(e) if isinstance(e,ValueError) else 'Proses gagal: '+type(e).__name__)
 if 'batch_result' in st.session_state: st.download_button('Unduh hasil terakhir',st.session_state['batch_result'],'finsight_predictions.csv','text/csv')
elif page=='Model Performance':
 heading('Performa model','Evaluasi deployment seed 42; berbeda dari agregat eksperimen historis.')
 phases=st.multiselect('Fase',['A','B'],default=['A','B']); split=st.radio('Pengujian',['test_id','test_ood'],horizontal=True); metric=st.selectbox('Metrik',metrics.columns[4:]); d=metrics[metrics.phase.isin(phases)&metrics.split.eq(split)].copy(); d['Konfigurasi']=d.model+' | '+d.phase+' | '+d['mode']; chart(px.bar(d.sort_values(metric),x=metric,y='Konfigurasi',color='phase',orientation='h',height=max(400,len(d)*24))); st.dataframe(d,hide_index=True); st.download_button('Unduh metrik deployment',metrics.to_csv(index=False),'deployment_metrics.csv'); st.subheader('Hasil historis (bukan deployment)'); st.dataframe(csv('tabel_hasil_akhir.csv'),hide_index=True); st.dataframe(csv('perbandingan_fase_A_vs_B.csv'),hide_index=True)
elif page=='Dataset Explorer':
 heading('Eksplorasi dataset','Statistik agregat; ulasan mentah dan ID pengguna tidak disertakan.')
 d=dist(); apps=st.multiselect('Aplikasi',sorted(d.app.unique()),default=sorted(d.app.unique())); sources=st.multiselect('Sumber label',sorted(d.sumber.unique()),default=sorted(d.sumber.unique())); d=d[d.app.isin(apps)&d.sumber.isin(sources)]; chart(px.bar(d.groupby(['Aspek','Sentimen'],as_index=False).jumlah.sum(),x='Aspek',y='jumlah',color='Sentimen',color_discrete_map=dict(zip(polarity,colors)))); st.dataframe(d,hide_index=True); st.subheader('Sebaran label gold per split'); st.dataframe(csv('sebaran_label_latih.csv'),hide_index=True); st.subheader('Ringkasan pembersihan'); st.dataframe(csv('ringkasan_data.csv'),hide_index=True)
else:
 heading('Metodologi','Dokumentasi eksperimen dan validasi artefak')
 st.markdown('**Alur:** Scraping → Pembersihan → Anotasi → Training → Evaluasi ID/OOD → Ekspor → Inferensi.\n\nFase A menggunakan gold label; Fase B menggunakan gold dan pseudo-label. Enam aspek masing-masing memiliki empat kelas: tidak dibahas, negatif, netral, positif. F1 pasangan mengukur kecocokan aspek dan polaritas; akurasi dapat tinggi akibat dominasi kelas tidak dibahas.')
 st.dataframe(pd.DataFrame([{'Model':m['name'],'Fase':m['phase'],'Mode':m['mode'],'Seed':m['seed'],'ID':m['artifact_id']} for m in registry]),hide_index=True); st.subheader('Laporan validasi Colab'); st.dataframe(csv('validation_report.csv'),hide_index=True); st.caption('Smoke test historis, bukan hasil inferensi sesi ini.'); st.dataframe(csv('smoke_predictions.csv'),hide_index=True)
