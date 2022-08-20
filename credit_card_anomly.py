# kullanılan kütüphanler
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




# Veriseti
df = pd.read_csv("./data/creditcard.csv")
df_class = df["Class"]
del df["Class"]
del df["Time"]



# kolonlar arasındaki ölçek farklılıklarını yok etmek için standartlaştırma işlemi
# PCA iyi çalışması için ölçeklendirme yapılmalı
scaler = StandardScaler()
df_scaler = scaler.fit_transform(df)
df = pd.DataFrame(df_scaler)


# PCA = küçük bir bilgi kaybını göze alarak boyut indirgemek için de kullanılabilir
pca = PCA(n_components=27)
df_pca = pd.DataFrame(pca.fit_transform(df))
print(np.cumsum(pca.explained_variance_ratio_))


# PCA uygulanan verisetini eski haline döndürme 
pca_convert_df = pd.DataFrame(pca.inverse_transform(df_pca))
# Geri çevirme işlemi yaparken bazı hatalar meydana gelir 
# bu hatalar anomaly verilerde daha çok olacaktır 
# bu sayede anomali veriler daha kolay tespit edilebilir


# oluşan iki veriseti arasındaki ortamala hataların karesi fonsiyonu
def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis = 1)
    loss = pd.Series(data=loss, index=df_original.index)
    return loss


# Satırların MSE değerleri
scores = get_anomaly_scores(df, pca_convert_df)
scores.plot(color="red")
plt.ylabel("Mean Squared Error", fontsize=14, color="green")
plt.title("scores", fontsize=14, color="green")



df_score = pd.DataFrame(scores, columns=["Score"])
pca_convert_df["Score"] = df_score

# oluşan hataların belli bir eşit değerden sonraki verilerin indexlerini bulma
def  threshold (number):
    anomly_index =[]
    for thres in pca_convert_df["Score"]:
        if thres >=number:
            index_no = pca_convert_df[pca_convert_df["Score"] == thres].index
            anomly_index.append(index_no[0])
    return anomly_index
df["Class"] =df_class





#! Karışıklık Matrisi (Confusion Matrix)

thres =14.5
pred_anomaly = df.iloc[threshold(thres)]["Class"].value_counts()
print(pred_anomaly)
tp = pred_anomaly[1]
fp = pred_anomaly[0]
fn = df["Class"].value_counts()[1] - tp
tn = len(df) - tp - fp - fn



class Confusion_matrix():
    def correct_classification_rate (tp,fp):
        return print("Precision: Doğru sınıflandırılan verilerin oranını verir.\n\tTP / (TP + FP) = %",(tp/(tp+fp)*100))
    def just_positive_classification_rate (tp,fn):
        return print("Recall: Sadece pozitif değerlerden doğru sınıflandırılanların oranını verir.\n\tTP / (TP + FN) = %",(tp/(tp+fn)*100))

Confusion_matrix.correct_classification_rate(tp,fp)
Confusion_matrix.just_positive_classification_rate(tp,fn)


confusion_matrix_ar =[[fp,fn],[tp,tn]] 

labels = [f"True Negative \n{tn}",
          f"False Negative \n{fn}",
          f"False Positive \n{fp}",
          f"True Positive \n{tp}"]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap( confusion_matrix_ar,annot=labels,fmt="")
plt.ylabel('Predicted', fontsize=14, color="green")
plt.xlabel('Actual', fontsize=14, color="green")
plt.savefig("confusion_matrix.png")
































