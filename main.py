import streamlit as st
import MLP
import ridge
import Linear
import RandomForestRegressor
import base64

st.title('求人原稿に対する応募数の予測')

uploaded_file=st.file_uploader("テストデータファイルをアップロード", type='csv')
option=st.selectbox("モデル選択",('多層パーセプトロン', 'リッジ回帰','線形回帰','ランダムフォレスト'))
a=0

if uploaded_file is not None and option == '多層パーセプトロン':
    ans=MLP.reg(uploaded_file)
    a=1
elif uploaded_file is not None and option =='リッジ回帰':
    ans=ridge.rid(uploaded_file)
    a=1
elif uploaded_file is not None and option == '線形回帰':
    ans=Linear.LR(uploaded_file)
    a=1
elif uploaded_file is not None and option == 'ランダムフォレスト':
    ans=RandomForestRegressor.RFR(uploaded_file)
    a=1

if a==1:
    st.write('予測結果')
    st.dataframe(ans)
    csv = ans.to_csv(index=False)  
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="result.csv">download</a>'
    st.markdown(f"予測データをダウンロードする {href}", unsafe_allow_html=True)
