import pandas as pd    # dataframele ilgili analiz yapabilmemizi sağlar. 
import numpy as np     # vektörleri içinde tutar. Matematiksel işlemler için kullanırız.
import seaborn as sns            # görselleştirme için
import matplotlib.pyplot as plt  # görselleştirme için
from matplotlib.colors import ListedColormap      # çıkan sonuçlarımızı görselleştirmek için kullanıcaz

from sklearn.preprocessing import StandardScaler         # standardization yapmak için kullanıcaz.
from sklearn.model_selection import train_test_split, GridSearchCV #GridSearc'i KNN'in best değerlerini bulurken kullanıcaz.
from sklearn.metrics import accuracy_score, confusion_matrix   # sonuçları değerlendirmek ve nedenini görmek için bunları kullanıyoruz.
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor # temel algoritmamız zaten neighbors üzerine  # Outlierlerimizi detect edicez bununla
from sklearn.decomposition import PCA

# warning library 
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("file:///C:/Users/asaro/Downloads/data.csv")
data.drop(['Unnamed: 32','id'],inplace=True,axis=1)  # kendisine eşitlemesi için inplace true dedik, y ekseni yani columnda silmesi için 1 dedik

data = data.rename(columns={"diagnosis":"target"})    # diagnosis olan sütun ismini target yaptık.

sns.countplot(data["target"])    # target classlabel larımızı görsellestirmek-hesaplamak için # maviler kötü, turuncular iyi huylu
print(data.target.value_counts())   # üstteki grafikteki değerleri tam olarak yazıyor.

data["target"] = [1 if i.strip()=='M' else 0 for i in data.target] #strip içinde bosluk varsa da kaldırıyor M için. Kötü huyluysa yani M ise 1 yaptık. String veriyi hem train için hem de kullanabilmek için numeric yaptık. 
 
print(len(data))     # datada kaç semple olduguna baktık
print(data.head())   # datanın ilk 5 satırını görürüz
print("Data Shape: ",data.shape)

data.info()
describe = data.describe()  

# describede veriler arası değerlerde çok fark oldugu için STANDARDİZATİON yapmamız lazım.
# missing valuelara bakıcaz datadan. cell'in boş veya 0 olması missing value oldugu anlamına gelebilr. 0 olan değerlerde domaine bağlı olarak belirli değerin altı threshold olsun denmiş olabilir. Bu datada missing value = None

# %% EDA

# correlation
corr_matrix = data.corr()  # categorical veriler olsa dışta bırakırdı ama zaten bizim datada numerical veriler var
sns.clustermap(corr_matrix, annot=True, fmt=".2f")  # korelasyonun görselleştirilip daha kolay anlaşılmasını sağlıcaz. # annot-true değerlerin gözükmesini sağlar # .2f ile değerlerin ilk 2 hanesini görücez
plt.title("Correlation Between Features")
plt.show()

# threshold yapıcaz
threshold = 0.75 
filtre = np.abs(corr_matrix["target"]) > threshold # target
corr_features = corr_matrix.columns[filtre].tolist()     # korelasyon matrisinin sütunlarını alıp filtreledik ve listeye cevirdik.
sns.clustermap(data[corr_features].corr(), annot=True, fmt=".2f")  # datamıza corr_featuresi uygulayıp korelasyona bakıcaz.
plt.title("Correlation Between Features w Corr Threshold 0.75") 

"""
there some correlated features
"""

# BOX PLOT
data_melted = pd.melt(data, id_vars = "target",        # id vars target oluyo çünkü boxplotta görselleştirirken iki farklı classımız oldugu için iki farklı class şeklinde görsellestiricez.
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)  # classlara ayırmak istedigimiz için target yazıyoruz, böylece ayrılır.
plt.xticks(rotation = 90) # featureslerin isimleri 90 derece dik olsun diye yapıyoruz
plt.show()


"""
standardization-normalization
"""

# pair plot  --- numeric veriler için kullanabileceğimiz en etkili yöntemlerden biri

sns.pairplot(data[corr_features],diag_kind="kde",markers="+",hue="target") # bize histogram olarak göstermesini istiyoruz(kde ile) # markers noktaları kasteder # target da iki classı ifade eder
plt.show()           # üstteki koddaki mavi ve turuncu noktalar classlarımız   

# HİSTOGRAMLARA BAKARAK DAĞILIMLARI YORUMLAYABİLİRİZ 
# Mavilerin kuyruğu sağa doğru uzanıyosa pozitif skewness diye adlandırırız.
# GAOS DAĞILIMI : insanların boyları gibi olabilir. belirli mean, standart sapması var. mean'in etrafında iki taraf simetriktir.
# Positive Skewness : Gelir dağılımı olabilir. Mean-Medyan-Mod değerleri grafikte sağdan sola doğrudur.
# Negative Skewness : Öğrenci notları olabilir. Mean-Medyan-Mod değerleri grafikte soldan sağa doğrudur.

# skew(); 1den büyükse pozitif, -1 den küçükse negatiftir.

## Outlier Detection yaparken olan skewnessi handle edebilecek outlier detection yöntemi kullanmak lazım. İkinci olarak da skewnessliği düzelterek normal dağılıma çevirmemiz lazım.

# Global ve local outlier çeşidi var. Local olan normalde veri kümesinde olması gerekip de herhangi sebepten bulunmayan demek.
# Biz DENSİTY BASED OUTLIER DETECTION SYSTEM kullanıcaz. Bu sistem içinden de LOCAL OUTLIER FACTOR(LOF) kullanıcaz. LOF, bir noktanın yerel yoğunluğunu karşılaştırmak demek. Bunu KNN ile yapıcaz.
# Herhangi bir noktanın outlier olup olmadıgı LOFa>1 ise outlier, değilse outlier değildir.

# %% OUTLIER

y = data.target            # y= target variablelarımız
x = data.drop(["target"],axis=1)   # x= targetı çıkarınca geriye kalanlar  # featurleri tuttugumuz değişken
columns = x.columns.tolist()          # datasetteki columnları columns değişkenine atadık.

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)    # y_predde oluşan -1ler outlier, 1ler inlier değerlerdir.
# normalde machine learningde y ile ypred karsılastırılır. Burdaki farklı ama; y targetları, ypred ise outlier olup olmadıgına bakıyor.
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()     # değerler - olacak negative outlier factorden dolayı. o değerleri + olarak düşünebiliriz.
outlier_score["score"] = X_score   # üstteki x_score u çalıstırmazsa diye dataframe olusturup öyle calıstırdık
# thresholdu 1.5 yaparsak 30 civarı outlier olur. dataset zaten 569, 30 cok olur bakıcaz.

# threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()
plt.figure()

plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color="b",s=50,label="Outliers")
plt.scatter(x.iloc[:,0],x.iloc[:,1], color="k", s=3, label="Data Points" )  # tüm satırları ve 0.columnu kullanıcaz demek ; tüm satırları ve 1. sütunu al, color, size, label

radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())   # normalizasyon yaparak bu değeri radiusa eşitledik. Radius büyükse outlier olma durumu artıyor.
outlier_score["radius"] = radius  # üstteki kodla göremezsek diye dataframe olusturup outlier score tablosuna attık
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none",label="Outlier Scores")  # görselleştirmek için yapıyoruz outlierleri
plt.legend()  # labelların görünebilmesi için gerekli bu 
plt.show()    # uzak olupda outlier gözükmeyen küçük çemberdekilerin sebebi resmin 2 boyutlu olması. Diğer taraflardan da görmek lazım

# DROP OUTLIERS
x= x.drop(outlier_index)  # indexe göre çıkartacağımız için outlier index aldık
y= y.drop(outlier_index).values   # y'nin type ını arraya çeviriyoruz.  x hala dataframe olarak kalıyor.

# %% TRAIN TEST SPLIT

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)  # randomstate 42 olması shuffle yapıldıgında süreki aynı değeri alması için. Eğer traintestden önce yapılan shuffle değişik değerler alırsa knn modelimiz etkilenebilir. Accuracy değerinin ayarlanırken bunu sabitleyip diğer variablelara bakarız.
# veri boyutu onbinler kadar büyük olsaydı testsize'ı daha küçük değerler de seçeçbilirdik. genelde 0.2 seçilir

# %% STANDARDIZATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # xtraini fit edip yeni xtraine transform et dedik.
X_test = scaler.transform(X_test)  # önceden xtrain e göre eğitilmiş scalerimizi xtest e uyguluyoruz. XTEST'i yeni xtrain imizie göre standardize etmemiz gerektigi için bunu yaptık

X_train_df = pd.DataFrame(X_train, columns=columns)   # az önce görselleştiremediğimiz boxplotu görsellestirmek için bunu yaptık
X_train_df_describe = X_train_df.describe()  # mean'i 0 std'si 1 oldugunu görüyoruz. Bu da normalize edildigini gösterir.
X_train_df["target"] = Y_train

data_melted = pd.melt(X_train_df, id_vars = "target",        # id vars target oluyo çünkü boxplotta görselleştirirken iki farklı classımız oldugu için iki farklı class şeklinde görsellestiricez.
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)  # classlara ayırmak istedigimiz için target yazıyoruz, böylece ayrılır.
plt.xticks(rotation = 90) # featureslerin isimleri 90 derece dik olsun diye yapıyoruz
plt.show()

sns.pairplot(X_train_df[corr_features],diag_kind="kde",markers="+",hue="target") # bize histogram olarak göstermesini istiyoruz(kde ile) # markers noktaları kasteder # target da iki classı ifade eder
plt.show()    # mean 0 std 1 oldu ama şeklimiz skewness dan normal dağılıma geçti mi? HAYIR!! verilerin şekli oldugu için mean ve std ile alakası yoktur. Şu an bunu görücez.

## KNN KULLANMA AVANTAJLARI = training süreci olmadıgı için daha hızlı, iplement etmesi kolay, tune etmesi kolay(sadece k ve distance değerini ayarlıyoruz)
## KNN KULLANMA DEZAVANTAJLARI = outlierlere sensitive bi algoritma yani outlier varsa güzel calısmaz, bigdatada kötüdür yavaşlar ve kötü sonuç üretir(zaten tüm machine learningler bigdatada sıkıntı çıkarır o yüzden neural network kullanıyoruz), çok feature varsa sıkıntı cıkartır(bunun için pca ve nca kullanıcaz), feature scaling yapmak lazım yoksa etkilenir, inbalance datada sıkıntı çıkartır  

# %% Basic KNN Method

knn = KNeighborsClassifier(n_neighbors=2) # k değeri 2 verdik
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
acc = accuracy_score(Y_test,y_pred)
score = knn.score(X_test, Y_test)  # acc_score la hiçbir farkı yok
print("Score: ",score)
print("Confusion Matrix: ",cm)
print("Basic KNN Acc: ",acc)

#[108   1]  # 108 tanesini iyi huylu olarak dogru tahmin etmisiz, 1 tanesini yanlıs tahmin etmisiz.
 #[  7  55] # toplam 62 tane kötü huylu veri varmış 7 tanesine iyi demişiz yanlıs bilmisiz. 55 tanesine kötü demisiz kötü cıkmıs

# OVERFITTING = verileri ezberlemeye başlayarak yeni veriseti geldiğinde doğru tahminlerde bulunamıyor. (high variance oluşur)
# UNDERFITTING = verilere model oturtuyoruz ama bu model verileri ifade edemiyorsa underfitting olur. Yani öğrenemiyor. (high bias oluşur)
# GOOD BALANCE = Model güzel eğitildiği verisetine oturmuş ama verisetini ezberlememiştir. (low variance, low bias)

# %% choose best parameters

def KNN_Best_Params(x_train, x_test, y_train, y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors=k_range, weights=weight_options) # grid search için gereken parametreleri bir dict'e koymamız gerekiyordu onu yaptık

    knn = KNeighborsClassifier()  # içinde hiçbir parametreye dokunmuyoruz çünkü birazdan grid search yapıcaz.
    grid = GridSearchCV(knn,param_grid,cv=10, scoring="accuracy") #gridsearchte machine learning modeli olan knn'i kullan, parametre olarak da param_grid dict'ini kullan diyoruz; cross valudationu 10 kez yap, skor olarak da accuracyi kabul edicez
    grid.fit(x_train,y_train)
    
    print("Best Training Score: {} with parameters : {}".format(grid.best_score_,grid.best_params_))
    print()
    # en iyi parametrelere sahip oldugumuz için artık bu gridi test veriseti üzerinde kullanabiliriz
    
    knn=KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)      # istesek bu iki satırda classifier yaratmak zorunda değiliz istersek grid.fit() ile de yapabiliriz.
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)  # herhangi bir overfit-underfit var mı görmek için train verisetimizi de test ediyoruz. 

    cm_test = confusion_matrix(y_test,y_pred_test)
    cm_train = confusion_matrix(y_train,y_pred_train)

    acc_test = accuracy_score(y_test,y_pred_test)
    acc_train = accuracy_score(y_train,y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test,acc_train))
    print()
    print("Confusion Matrix Test : ",cm_test)
    print("Confusion Matrix Train : ",cm_train)

    return grid # en iyi parametrelere sahip grid variable ımız
    # methodumuz artık HAZIR
    
grid = KNN_Best_Params(X_train,X_test,Y_train,Y_test)

# Test Score: 0.9590643274853801, Train Score: 0.9773299748110831 ---- train testten cok cıkmıs yani ezberleme söz konusu (overfitting)
#  üsttekine göre test set error=%5,train set error=%3  # farkı aşağı yukarı yüzde 10 olsa kesin overfit diyebilirdik. şu an overfite doğru gidiyor.
# aradaki oran az olsa da ikisi de errorlar kötü olsaydı yüzde 20-30 gibi ; low variance var iyi derdik ama bu sefer de hata cok yani high bias sözkonusu 
# yüzde 1 train error, yüzde 1.5 test error vs olsa en ideali. Low variance and low bias
### best parametreleri bulmak için model complexitiy'i arttırdık bu sefer de high variance oldu bu overfite sebep oldu. Bunu engellemek için crossvalduation yaptık. ama yetmedi. İkiisnin arası olan yüzde 96 gibi bi degerde muhtemelen ideal olacaktır. Bunun için model complexitiy azaltıcaz ya da regularization yapıcaz 

## basic knn acc'e göre yüzde 1.5 arttı. Trainden biraz ödün vererek test i iyi hale getirdik


# %% PCA                     

"""
PCA'in overfitting-underfittingle alakası yok!! - Mümkün oldugu kadar bilgi tutarak verinin boyutunun azaltılmasını sağlar  - elimizde korelasyon matrisi varsa featureler arasındaki korelasyona göre nasıl ortadan kaldıracağımızı bilmiyorsak yine PCA kullanabiliriz
knn i görselleştirebiliyoruz pca ile; 30 boyuttan 2 boyuta indirerek eğitip nasıl çalıştığını görselleştiricez.
PCA'nın amacı eigen vector ve eigen value ları bulmak --- COVARIANS matrisinden bunları bulucaz
Eigen Value= büyüklük ; Eigen Vector = yeni feature space'imizin vektöreli
"""        


"""                                              PCA ÖRNEĞİ
x=[2.4,0.6,2.1,2,3,2.5,1.9,1.1,1.5,1.2]
y=[2.5,0.7,2.9,2.2,3.0,2.3,2.0,1.1,1.6,0.8]

x=np.array(x)
y=np.array(y)
plt.scatter(x,y)
#bunu 0 merkeze çekicez şimdi
x_m = np.mean(x)
y_m = np.mean(y)

x=x - x_m
y=y - y_m
plt.scatter(x,y)   # 0 merkeze çektik.

c= np.cov(x,y)  # covarians matrisimizi yazdırdık
from numpy import  linalg as LA

w,v = LA.eig(np.cov(x,y))
p1= v[:,1]
p2= v[:,0]
plt.plot([-2*p1[0],2*p1[0]],[-2*p1[1],2*p1[1]])  # normalde 0,p1[0];0,p1[1] diyeydi ama cok kısa oldugu için çizgiyi uzattık böyle
plt.plot([0,p2[0]],[0,p2[1]])      # bu ikinci plandaki çizgi oldugu için bilerek uzatmadım 2yle çarpıp
"""

#%% PCA2

# pca yapmadan önce verimizi scale etmemiz yani standardize etmemiz gerekiyor ; çünkü pca class labellara ihtiyaç duymuyor bu yüzden train test diye ayırmadan direkt x inputlarını kullanıcaz

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)      # pca'in 2 bileşenini kullanıcaz demek ; x'in içinde 30 feature var bunu 2 ye düşürücez dedik burda
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)   # buna bakabilmek için bir dataframein içine atıcaz

pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2"])
pca_data["target"] = y       # pca datamıza target ekliyoruz burda, görselleştirme yaparken kullanabilmek için 
sns.scatterplot(x="p1",y="p2",hue="target",data=pca_data)   # classlarımızı farklı renklerde görmek istiyoruz dedik hue=target ile
plt.title("PCA : p1 vs p2")   # PCA'da kullandıgımız 2 temel bileşenimizi görselleştirmek istiyoruz

# burada DİMENSİON REDUCTİON yaptık , 30 boyutu 2 boyuta indirdik.
# şimdi 2 boyutlu verimizi kullanarak knn ile training sınıflandırma yapıcaz

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca,y,test_size=0.3,random_state=42)

grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)

# Yanlıs tahmin ettigimiz değerleri görselleştiricez şimdi, bu sayede algoritmamız nasıl karar veriyor onu anlarız


# VISUALIZE

cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])    # bu iki satırda renkleri seçip iki variableye eşitledik

h= .05 # step size in the mesh      # 0.05 diye step size seçtik
X = X_reduced_pca                               # birazdan cok kullanacagımız için karısıklık olmasın diye x'e eşitledik
x_min,x_max = X[:,0].min() - 1 , X[:,0].max() +1
y_min,y_max = X[:,1].min() - 1 , X[:,1].max() +1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),          # bu 3 satırda en sonki p1vsp2 deki tablodaki y ve x sınırlarının max ve min değerlerini alarak 0.05 adımlarla bir meshgrid oluştur diyoruz
                  np.arange(y_min,y_max,h))           # bu meshgridi olusturma sebebimiz birazdan meshgrid içindeki her bir noktayı sınıflandırma algoritmamıza sokacak olmamız

Z = grid_pca.predict(np.c_[xx.ravel(),yy.ravel()])  # prediction yapabilmek için önce xx-yy'yi düzleştirmemiz gerekiyordu ravel ile onu yaptık; c_ methodu ile bunları birleştirdik.  # üstte dediğimiz sınıflandırma algoritmamıza burada sokuyoruz

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)  # Z'ye cmap lightdaki renkleri kullandırıyoruz.2farklı classımız var burda, zaten Znin içinde de iki farklı class oldugu için 2 farklı renk ortaya cıkar

# Plot also the training points
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,
            edgecolor='k',s=20)     # iki boyutlu kendi verimi bu plotun üstüne ekliyoruz, x_reduced_cpa dan gelen verimiz
plt.xlim(xx.min(),xx.max())    # eksenlerin boyunu ayarladık
plt.ylim(yy.min(),yy.max())     # eksenlerin boyunu ayarladık
plt.title("%i-Class classification (k = %i , weights = '%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights ))

#   NCA               - sınıflandırma performansını maksimize edecek şekilde input verilerinin doğrusal dönüşümü kullanarak mesafe metriğini öğrenmek
                    # nca ile ; rastgele bir distance metriği belirlemek yerine doğrusal dönüşümü bularak bu metriği nca algoritması kendisi öğreniyor

nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)   # componentslerimizi 2ye eşitledik yine
nca.fit(x_scaled,y)   # nca; unsupervised learning değildir bu yüzden fit işlemi yaparken y'ye yani target variableye ihtiyaç duyar
X_reduced_nca = nca.transform(x_scaled)

nca_data = pd.DataFrame(X_reduced_nca,columns=["p1","p2"])    # bir verisetinin dataframe ine yerleştirerek görselleştirme yapıcaz
nca_data["target"] = y                # datamıza targetı ekliyoruz
sns.scatterplot(x="p1",y="p2",hue="target",data=nca_data)       # hue ile targetıma göre sınıflandırma rengi ata dedik
plt.title("NCA: p1 vs p2")

## NCA in sınıflandırması-ayrıştırması PCA'dan daha iyi. veri kaybı daha az

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca,y,test_size=0.3,random_state=42)

grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)      # test score da yüzde 99 trainde yüzde 100 başarı getirdi.

# görselleştiriyoruz
cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange','darkblue'])    # bu iki satırda renkleri seçip iki variableye eşitledik

h= .2 # step size in the mesh      # 2 diye step size seçtik daha az kassın diye, zorlu bi model
X = X_reduced_nca                               # birazdan cok kullanacagımız için karısıklık olmasın diye x'e eşitledik
x_min,x_max = X[:,0].min() - 1 , X[:,0].max() +1
y_min,y_max = X[:,1].min() - 1 , X[:,1].max() +1
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),          # bu 3 satırda en sonki p1vsp2 deki tablodaki y ve x sınırlarının max ve min değerlerini alarak 0.05 adımlarla bir meshgrid oluştur diyoruz
                  np.arange(y_min,y_max,h))           # bu meshgridi olusturma sebebimiz birazdan meshgrid içindeki her bir noktayı sınıflandırma algoritmamıza sokacak olmamız

Z = grid_nca.predict(np.c_[xx.ravel(),yy.ravel()])  # prediction yapabilmek için önce xx-yy'yi düzleştirmemiz gerekiyordu ravel ile onu yaptık; c_ methodu ile bunları birleştirdik.  # üstte dediğimiz sınıflandırma algoritmamıza burada sokuyoruz

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)  # Z'ye cmap lightdaki renkleri kullandırıyoruz.2farklı classımız var burda, zaten Znin içinde de iki farklı class oldugu için 2 farklı renk ortaya cıkar

# Plot also the training points
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,
            edgecolor='k',s=20)     # iki boyutlu kendi verimi bu plotun üstüne ekliyoruz, x_reduced_cpa dan gelen verimiz
plt.xlim(xx.min(),xx.max())    # eksenlerin boyunu ayarladık
plt.ylim(yy.min(),yy.max())     # eksenlerin boyunu ayarladık
plt.title("%i-Class classification (k = %i , weights = '%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights ))


### yüzde 99 başarıda n_neighboru 1 çıktı. normalde 1 cıkması yüzde 100 overfit demek eğer testscore az cıksaydı. ama su an sorun yok

# %% find wrong decision       (yanlıs sınıflandırdıgımız veriyi görselleştiriyoruz)

knn= KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2",hue="Y_test_nca",data=test_data)

diff=np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label="Wrong Classified",edgecolor="green",s=20)


















