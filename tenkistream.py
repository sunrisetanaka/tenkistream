import requests
from bs4 import BeautifulSoup
import time
import re
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM,Dropout,concatenate
from keras.optimizers import Adam
import tensorflow as tf
from keras import Input
import warnings
warnings.simplefilter("ignore",FutureWarning)
import streamlit as st

prog=0
progress_bar=st.progress(prog)
subprog=0
subprog_bar=st.progress(subprog)
def main():
    st.title("Ki-ON-Stream")
    st.write("今日の気温を予測します")
    if st.button("予測を始める(絶対に連打しない)"):
        st.write("データ収集中")
        df=scrap_df()
        nowdf=scrap_df_now()
        st.write("データ確保")
        df=df.fillna(df.median())
        nowdf=nowdf.fillna(nowdf.median())
        st.write("学習中")
        mid,top,bot=ml(df,nowdf)
        st.success("最低気温:{}".format(bot))
        st.success("平均気温:{}".format(mid))
        st.success("最高気温:{}".format(top))
    else:
        st.write("待機中")

#kerasコールバック
class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        keys = list(logs.keys())
        global prog
        if prog<100:
            prog+=8
            progress_bar.progress(prog)
    
    def on_train_batch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        global subprog
        if subprog<100:
            subprog+=1
            subprog_bar.progress(subprog)
        elif subprog==100:
            subprog=0
            subprog_bar.progress(subprog)
    

#スクレイピングからデータフレームの作成まで
def scrap_df():
    global prog
    global subprog
    #日付取得
    dt_now=datetime.datetime.now()
    year_now=dt_now.year
    month_now=dt_now.month
    #データフレームの初期化
    df=pd.DataFrame()
    #順番に処理
    for y in range(year_now-20,year_now+1):
        for m in range(1,13):
            load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=61&block_no=47759&year={}&month={}&day=&view=".format(str(y),str(m))
            try:

                #データロード
                html=requests.get(load_url)
                soup=BeautifulSoup(html.content,"html.parser")
            
                #要素をリスト化
                monthdata=[]
                for td in soup.find_all("td"):
                    monthdata.append(td.text)
                
                #中身がない要素を削除
                monthdata=[i for i in monthdata if i!='']
                
                #urlから要素検索
                year=re.findall(r"[12]\d{3}",load_url)[0]
                month=re.findall(r"month=(0?[1-9]|1[0-2])",load_url)[0]
                #データを分ける
                monthdict={
                    "年":[],
                    "月":[],
                    "日":[],
                    "平均気温":[],
                    "最高気温":[],
                    "最低気温":[],
                    "日照時間":[]
                }
                #データを詰め込んでいく
                for i in monthdata[::21]:
                    monthdict["年"].append(year)
                    monthdict["月"].append(month)
                    monthdict["日"].append(i)
                #余計な文字削除
                dict=monthdict["日"]
                dict.remove("利用される方へ")
                
                #yearとmonth調整
                dict=monthdict["年"]
                dict.pop()
                dict=monthdict["月"]
                dict.pop()
                
                #他も詰め込む
                for i in monthdata[6::21]:
                    monthdict["平均気温"].append(i)
                
                for i in monthdata[7::21]:
                    monthdict["最高気温"].append(i)
                
                for i in monthdata[8::21]:
                    monthdict["最低気温"].append(i)
                
                for i in monthdata[16::21]:
                    monthdict["日照時間"].append(i)
                
                #データフレーム作成
                monthdf=pd.DataFrame(monthdict)
                
                #変な文字を消す
                param=["平均気温","最高気温","最低気温","日照時間"]
                
                for i in param:
                    monthdf[i]=monthdf[i].str.replace(")","")
                    monthdf[i]=monthdf[i].str.replace("]","")
                    monthdf[i]=monthdf[i].str.replace(" ","")
                
                #元のdfに追加
                df=pd.concat([df,monthdf])
    
                for i in param:
                    df[i]=pd.to_numeric(df[i],errors="coerce")
    
                #サーバー負荷軽減
                time.sleep(1)
                subprog+=8
                subprog_bar.progress(subprog)
                
            except:
                pass
        prog+=3
        progress_bar.progress(prog)
        subprog=0
        subprog_bar.progress(subprog)

            
    
        

    df=df.fillna(df.median())
    return df

#今日含めない30日のデータ収集
def scrap_df_now():

    #日付取得
    dt_now=datetime.datetime.now()
    year_now=dt_now.year
    month_now=dt_now.month
    date_now=dt_now.day

        #データを分ける
    monthdict={
        "年":[],
        "月":[],
        "日":[],
        "平均気温":[],
        "最高気温":[],
        "最低気温":[],
        "日照時間":[]
    }
    #30日に満たない場合前月のデータ参照する
    if date_now<=30:
        #もし1月なら
        if month_now<=1:
            load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=61&block_no=47759&year={}&month={}&day=&view=".format(str(year_now-1),str(12))

            html=requests.get(load_url)
            soup=BeautifulSoup(html.content,"html.parser")
            ifdata=[]
            for td in soup.find_all("td"):
                ifdata.append(td.text)
            
            #中身がない要素を削除
            ifdata=[i for i in ifdata if i!='']
            
            #urlから要素検索
            year=re.findall(r"[12]\d{3}",load_url)[0]
            month=re.findall(r"month=(0?[1-9]|1[0-2])",load_url)[0]
            

            #データを詰め込んでいく
            for i in ifdata[::21]:
                monthdict["年"].append(year)
                monthdict["月"].append(month)
                monthdict["日"].append(i)

            #データ加工(最後2つだけdelete)
            for i in range(2):
                monthdict["年"].pop()
                monthdict["月"].pop()
                monthdict["日"].pop()

            #他も詰め込む　(pop)でデータの加工も
            for i in ifdata[6::21]:
                monthdict["平均気温"].append(i)
                
            for i in ifdata[7::21]:
                monthdict["最高気温"].append(i)
            
            for i in ifdata[8::21]:
                monthdict["最低気温"].append(i)
                
            for i in ifdata[16::21]:
                monthdict["日照時間"].append(i)

            monthdict["平均気温"].pop()
            monthdict["最高気温"].pop()
            monthdict["最低気温"].pop()
            monthdict["日照時間"].pop()
            
        #1月以外
        else:
            load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=61&block_no=47759&year={}&month={}&day=&view=".format(str(year_now),str(month_now-1))

            html=requests.get(load_url)
            soup=BeautifulSoup(html.content,"html.parser")
            ifdata=[]
            for td in soup.find_all("td"):
                ifdata.append(td.text)
            
            #中身がない要素を削除
            ifdata=[i for i in ifdata if i!='']
            
            #urlから要素検索
            year=re.findall(r"[12]\d{3}",load_url)[0]
            month=re.findall(r"month=(0?[1-9]|1[0-2])",load_url)[0]
            #データを分ける

            #データを詰め込んでいく
            for i in ifdata[::21]:
                monthdict["年"].append(year)
                monthdict["月"].append(month)
                monthdict["日"].append(i)

            #データ加工(最後2つだけdelete)
            for i in range(2):
                monthdict["年"].pop()
                monthdict["月"].pop()
                monthdict["日"].pop()

            #他も詰め込む　(pop)でデータの加工も
            for i in ifdata[6::21]:
                monthdict["平均気温"].append(i)
                
            for i in ifdata[7::21]:
                monthdict["最高気温"].append(i)
            
            for i in ifdata[8::21]:
                monthdict["最低気温"].append(i)
                
            for i in ifdata[16::21]:
                monthdict["日照時間"].append(i)

            monthdict["平均気温"].pop()
            monthdict["最高気温"].pop()
            monthdict["最低気温"].pop()
            monthdict["日照時間"].pop()

            time.sleep(1)

    #その月のデータ
    load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=61&block_no=47759&year={}&month={}&day=&view=".format(str(year_now),str(month_now))
    #データロード
    html=requests.get(load_url)
    soup=BeautifulSoup(html.content,"html.parser")

    #要素をリスト化
    monthdata=[]
    for td in soup.find_all("td"):
        monthdata.append(td.text)
    
    #中身がない要素を削除
    monthdata=[i for i in monthdata if i!='']
    
    #urlから要素検索
    year=re.findall(r"[12]\d{3}",load_url)[0]
    month=re.findall(r"month=(0?[1-9]|1[0-2])",load_url)[0]

    #データを詰め込んでいく
    for i in monthdata[::21]:
        monthdict["年"].append(year)
        monthdict["月"].append(month)
        monthdict["日"].append(i)

    #データ加工(最後2つだけdelete)
    for i in range(2):
        monthdict["年"].pop()
        monthdict["月"].pop()
        monthdict["日"].pop()

    #他も詰め込む　(pop)でデータの加工も
    for i in monthdata[6::21]:
        monthdict["平均気温"].append(i)
        
    for i in monthdata[7::21]:
        monthdict["最高気温"].append(i)
    
    for i in monthdata[8::21]:
        monthdict["最低気温"].append(i)
        
    for i in monthdata[16::21]:
        monthdict["日照時間"].append(i)

    monthdict["平均気温"].pop()
    monthdict["最高気温"].pop()
    monthdict["最低気温"].pop()
    monthdict["日照時間"].pop()
    
    
    #データフレーム作成
    nowdf=pd.DataFrame(monthdict)
    #変な文字を消す
    param=["平均気温","最高気温","最低気温","日照時間"]
    
    for i in param:
        nowdf[i]=nowdf[i].str.replace(")","")
        nowdf[i]=nowdf[i].str.replace("]","")
        nowdf[i]=nowdf[i].str.replace(" ","")

    #数値化
    for i in param:
        nowdf[i]=pd.to_numeric(nowdf[i],errors="coerce")

    #30日のデータにする
    nowdf=nowdf.drop(nowdf.index[:-30])
    print(nowdf)

    nowdf=nowdf.fillna(nowdf.median())
    return nowdf

#学習と予測
def ml(df,nowdf):
    look_back=30
    window_size=30
    #入力を分ける
    input_data1=df["平均気温"].values.astype(float)
    input_data2=df["最高気温"].values.astype(float)
    input_data3=df["最低気温"].values.astype(float)
    input_data4=df["日照時間"].values.astype(float)
    
    #スケールの正規化

    norm_scale1=input_data1.max()
    input_data1/=norm_scale1

    norm_scale2=input_data2.max()
    input_data2/=norm_scale1

    norm_scale3=input_data3.max()
    input_data3/=norm_scale1

    norm_scale4=input_data4.max()
    input_data4/=norm_scale1

    X1,X2,X3,X4,y1,y2,y3=[],[],[],[],[],[],[]
    for i in range(len(input_data1)-look_back):
        X1.append(input_data1[i:i+look_back])
        X2.append(input_data2[i:i+look_back])
        X3.append(input_data3[i:i+look_back])
        X4.append(input_data4[i:i+look_back])
        y1.append(input_data1[i+look_back])
        y2.append(input_data2[i+look_back])
        y3.append(input_data3[i+look_back])

    X1=np.array(X1)
    X2=np.array(X2)
    X3=np.array(X3)
    X4=np.array(X4)
    y1=np.array(y1)
    y2=np.array(y2)
    y3=np.array(y3)

    #データ成型
    X1=X1.reshape(X1.shape[0],X1.shape[1],1)
    X2=X2.reshape(X2.shape[0],X2.shape[1],1)
    X3=X3.reshape(X3.shape[0],X3.shape[1],1)
    X4=X4.reshape(X4.shape[0],X4.shape[1],1)

    #訓練データとテストデータに分ける
    X1_train,X1_test,X2_train,X2_test,X3_train,X3_test,X4_train,X4_test,y1_train,y1_test,y2_train,y2_test,y3_train,y3_test=train_test_split(X1,X2,X3,X4,y1,y2,y3,test_size=0.3,shuffle=False,random_state=1)


    #ネットワーク構築
    input_a=Input(shape=(window_size,1),name="input_a")
    lstm_a=LSTM(100,return_sequences=True)(input_a)
    drop_a=Dropout(0.2)(lstm_a)
    lstm_a2=LSTM(100)(drop_a)
    drop_a2=Dropout(0.2)(lstm_a2)

    input_b=Input(shape=(window_size,1),name="input_b")
    lstm_b=LSTM(100,return_sequences=True)(input_b)
    drop_b=Dropout(0.2)(lstm_b)
    lstm_b2=LSTM(100)(drop_b)
    drop_b2=Dropout(0.2)(lstm_b2)

    input_c=Input(shape=(window_size,1),name="input_c")
    lstm_c=LSTM(100,return_sequences=True)(input_c)
    drop_c=Dropout(0.2)(lstm_c)
    lstm_c2=LSTM(100)(drop_c)
    drop_c2=Dropout(0.2)(lstm_c2)
    
    input_d=Input(shape=(window_size,1),name="input_d")
    lstm_d=LSTM(100,return_sequences=True)(input_d)
    drop_d=Dropout(0.2)(lstm_d)
    lstm_d2=LSTM(100)(drop_d)
    drop_d2=Dropout(0.2)(lstm_d2)

    concat=concatenate([drop_a2,drop_b2,drop_c2,drop_d2],axis=-1)

    output1=Dense(1,name="output1")(concat)
    output2=Dense(1,name="output2")(concat)
    output3=Dense(1,name="output3")(concat)

    
    model=Model([input_a,input_b,input_c,input_d],[output1,output2,output3])

    #コンパイル
    model.compile(loss="mean_squared_error",optimizer=Adam(),metrics=["mae"])
    #学習
    batch_size=52
    n_epochs=5

    history = LossHistory()
    hist=model.fit({"input_a":X1_train,"input_b":X2_train,"input_c":X3_train,"input_d":X4_train},{"output1":y1_train,"output2":y2_train,"output3":y3_train}
    ,epochs=n_epochs,validation_data=({"input_a":X1_test,"input_b":X2_test,"input_c":X3_test,"input_d":X4_test},{"output1":y1_test,"output2":y2_test,"output3":y3_test}),verbose=1,batch_size=batch_size,callbacks=[history])

    #学習の進捗を画像保存
    plt.plot(hist.history["loss"],label="train set")
    plt.plot(hist.history["val_loss"],label="test set")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("model_loss.jpg")
    plt.show()

    #入力を分ける
    input_data1=nowdf["平均気温"].values.astype(float)
    input_data2=nowdf["最高気温"].values.astype(float)
    input_data3=nowdf["最低気温"].values.astype(float)
    input_data4=nowdf["日照時間"].values.astype(float)
    
    #スケールの正規化
    input_data1/=norm_scale1
    input_data2/=norm_scale1
    input_data3/=norm_scale1
    input_data4/=norm_scale1

    nowX1=np.array(input_data1)
    nowX2=np.array(input_data2)
    nowX3=np.array(input_data3)
    nowX4=np.array(input_data4)

    #データ成型
    nowX1=nowX1.reshape(1,window_size,1)
    nowX2=nowX2.reshape(1,window_size,1)
    nowX3=nowX3.reshape(1,window_size,1)
    nowX4=nowX4.reshape(1,window_size,1)
    
    #予測
    pred=model.predict({"input_a":nowX1,"input_b":nowX2,"input_c":nowX3,"input_d":nowX4})
    mid=pred[0]
    mid*=norm_scale1
    top=pred[1]
    top*=norm_scale1
    bot=pred[2]
    bot*=norm_scale1
    print("平均気温:{} 最高気温:{} 最低気温:{}".format(mid,top,bot))
    



    

    


    





if __name__ == "__main__":
    main()