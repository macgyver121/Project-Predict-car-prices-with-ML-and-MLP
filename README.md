# Introduction
จุดประสงค์การศึกษาครั้งนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep learning คณะสถิติประยุกต์ สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล(DADS) สถาบันบัณฑิตพัฒนบริหารศาสตร์ โดยมี ผศ.ดร.ฐิติรัตน์ ศิริบวรรัตนกุล เป็นผู้สอน

โดยทำการศึกษาเปรียบเทียบประสิทธิภาพการทำงานระหว่าง traditional machine learning กับ MLP โดยใช้ชุดข้อมูลตัวอย่างเป็น ราคารถยนต์มือสองในสิงคโปร์ปี2021 จาก https://www.kaggle.com/datasets/jiantay33/singapore-used-car

# Data
## Data source 
ที่สิงคโปร์ ตลาดรถมือสองมีปัจจัยหลายอย่างที่ส่งผลกับหลายค่า เนื่องจากการมีการควบคุมกฎหมาย ในcolumnต่างๆคือปัจจัยที่ส่งผลต่อการเลือกซื้อรถยนต์ 1 คันในสิงคโปร์

1. Brand: ยี่ห้อของรถยนต์ในสิงคโปร์
2. Type: โมเดลชนิดของรถยนต์ในสิงคโปร์
3. Reg_date: วันที่รถยนต์ถูกลงทะเบียน
4. Coe_left: ใบรับรองสิทธิที่เหลือในการครอบครองรถยนต์ในสิงค์โปร์ เนื่องจากกฎหมายจำกัดจำนวนรถ โดยเฉลี่ยต่อคันจะมีอายุการใช้งาน 10 ปี 
5. Dep: ค่าเสื่อมราคาตามเวลาของการใช้รถยนต์
6. Mileage: ตัวบ่งบอกว่ารถคันนี้ได้ถูกใช้วิ่งมากี่ไมล์
7. Road_Tax: ภาษีที่ผู้ใช้รถในสิงคโปร์ต้องจ่ายไปยังส่วนกลางหรือรัฐบาลเพื่อใช้บำรุงหรือพัฒนาโครงสร้างการจราจร
8. Dereg_value: เงินที่จะได้คืนหลังจากทำการเพิกถอนสิทธิครอบคลองรถยนต์
9. COE: ราคาประมูลสิทธิครอบครองรถยนต์ในสิงคโปร์
10. Engine_Cap: ขนาดความจุของเครื่องยนต์รถแต่ละโมเดล
11. Curb_weight: น้ำหนักของรถยนต์โดยเฉลี่ย
12. Manufactured: วันที่รถยนต์ถูกผลิต
13. Transmission: รูปแบบการทำงานของเกียร์
14. OMV: ราคารถยนต์เปล่าจากโรงงาน ก่อนบวกภาษี
15. ARF: ค่าธรรมเรียมการจดทะเบียนเพิ่มเติม จะถูกเรียกเก็บเมื่อลงทะเบียนยานภาหนะในสิงคโปร์
16. Power: กำลังของรถยนต์
17. No.of_Owners: จำนวนของผู้ที่เคยถือครองรถยนต์คันนี้
18. Price: ราคาของรถยนต์ 

## Cleansing data

เราทำการ clean ข้อมูลผ่าน application ชื่อ Openrefine โดยการกรุ๊ปชื่อตัวแปรให้เข้าใจง่าย เช่น Brand ของรถยนต์, เปลี่ยน type ของตัวแปร
ให้เป็น numerical ยิ่งกว่านั้นเราใช้ python ทำการกำจัด null และ coloumn ที่ไม่จำเป็นออก

```
df = df.replace('N.A', np.NaN)
df = df.replace('N.A.', np.NaN)
del df['Unnamed: 18']
df = df.dropna()
```

info ของข้อมูลหลัง clean เสร็จ

```
df.info()
```
![image](https://user-images.githubusercontent.com/85028821/189877388-1480e1ac-0fa8-4f41-882c-222a927e96eb.png)

# EDA
พล็อตกราฟเพื่อดูการกระจายตัวของข้อมูล ในแต่ละปัจจัยที่มีโอกาศส่งผลต่อราคารถมือสองในสิงคโปร์

```
def set_style(ax):
    sns.despine(ax=ax, left=True)
    ax.grid(axis='y', linewidth=0.3, color='black')
    
# Black - Aqua Blue - Pink
colors = ["#09101F","#72DDF7", '#F7AEF8']

def hist(df, x, ax, main_color=colors[1], second_color=colors[0], bins=30):
    
    sns.histplot(data=df, x=x, bins=bins, ax=ax,
                 kde=True, color=main_color,
                 edgecolor=second_color, line_kws={"linestyle":'--'}, linewidth=3
                )
    ax.lines[0].set_color(second_color)
    set_style(ax)
    
    ax.set_xlabel(x.replace("_", " ").capitalize(), fontsize="x-large")
    ax.set_ylabel("")
    
cols = ['Coe_left(yrs)', 'Dep($)', 'Mileage(mile)', 'Road Tax($)', 'Dereg Value($)', 'COE($)', 'Engine Cap(cc)',
       'Curb Weight(kg)', 'Manufactured(yrs)', 'OMV($)', 'ARF($)', 'Power', 'No. of Owners', 'Price($)' ]

fig, axs = plt.subplots(4, 4, figsize=(30, 25))

for i, col in enumerate(cols):
    
    row_index = i // 4
    col_index = i % 4
    
    hist(df, col, axs[row_index][col_index])

fig.suptitle("Histograms of numeric columns of the Dataset", fontsize="xx-large", y=0.92)
    
plt.show()
```

![graph1](https://user-images.githubusercontent.com/85028821/189380224-0e4db924-6f53-4bcb-b83a-a7bec6f8eaed.png)

1. Dereg value, OMV และ Arf มีทิศทางไปในทิศทางเดียวกับราคารถยนต์ ซึ่งอาจแปลได้ว่า ถ้าเราทราบค่าปัจจัยทั้ง 3 นี้ อาจสามารถประเมินราคารถยนต์มือสองได้
2. จากกราฟ Coe left จะเห็นว่าในช่วงที่ใบ Coe เหลือ 6-10 ปี กราฟมีแนวโน้มเพิ่มสูงขึ้น แปลอีกนัยว่า รถที่นำมาขายในตลาดมือสองนั้น พึ่งมีการต่อใบ Coe มาได้ประมาณ 4 ปี และรถที่มีใบ Coe เหลืออายุน้อยกว่า 4 ปีมีปริมาณน้อย และ จากกราฟ Manufacturing ปีของรถยนต์ จะเห็นได้ว่า กราฟมีลักษณเว้าลงในช่วง 10 ปี ซึ่งนั้นคือช่วงที่ต้องต่ออายุใบ Coe จึงแปลได้ว่า รถที่พึ่งต่ออายุ Coe มาจมีแนวโน้มที่จะนำออกมาขายน้อย
	โดยจาก 2 กราฟที่อธิบายด้านบน จะเห็นว่า มีความสัมพันธ์กันในช่วงอายุรถ 6-10 นั้นคือ ช่วงที่ใบCoe กำลังจะหมดอายุ หรือก็คือ Coe left เหลือ 1-3 ปี จะมีปริมาณรถที่ถูกนำมาขายน้อย พอมาในช่วงอายุรถ10-13ปี คือช่วงที่รถพึ่งมีการต่อใบ Coe ได้ประมาณ 1-3 ปี หรือก็คือ Coe left เหลือ 6-9 ปี จะมีปริมาณรถที่ถูกนำมาขายมาก


```
def count(df, x, ax, main_color=colors[2], second_color=colors[0]):
    
    ax.bar(df[x].value_counts().index, df[x].value_counts().values,
           color=main_color, edgecolor=second_color, linewidth=3)
    
    set_style(ax)
    
    ax.set_xlabel(x.replace("_", " ").capitalize(), fontsize="x-large")
    ax.set_ylabel("")

cols = ['Brand', 'Type', 'Transmission']

fig, ax = plt.subplots(1, 3, figsize=(30, 7))

for i, col in enumerate(cols):
    count(df, col, ax[i])
    
fig.suptitle("Count of values for categorical columns", size="xx-large")

plt.show()
```
![graph2](https://user-images.githubusercontent.com/85028821/189382343-8674757a-6be6-4e62-a4b0-6ef75cf954c2.png)

จากการกระจายตัวของข้อมูลประเภทที่ไม่ใช่ตัวเลข จะเห็นได้ว่า  5 อันดับแรก ของแบรนด์รถที่ถูกนำมาขายในตลาดมือสอง ก็คือ Mercedes-Benz, Honda, Toyota, BMW และ Mazda ส่วนประเภทของรถ ก็คือ SUV, Luxury Sedan, Mid-Sized Sedan, Hatchback และ Sports car ในส่วนของชนิดเกียร์ของรถ โดยประมาณ 98% รถที่นำมาขายเป็นรถที่ใช้เกียร์ Auto

```
df['Brand'].value_counts().head()

df['Type'].value_counts().head()

df['Transmission'].value_counts().head()
```
![top5-brand](https://user-images.githubusercontent.com/97573140/189682716-0ce08a7a-f6f2-4feb-9ae2-50c78f1302f8.png)
![top5-type](https://user-images.githubusercontent.com/97573140/189682747-f97e4b88-cc46-4b8e-84a2-adb61cfd0cea.png)

## Correlation
ต่อมาเราจะดูปัจจัยต่างๆที่มีผลต่อราคารถยนต์มือ2ในสิงคโปร์โดยสังเกตุจากความสัมพันธ์ระหว่างตัวแปรต่างๆข้างต้นกับราคารถเราจะได้กราฟความสัมพันธ์ดังนี้
```
def scatter(df, x, y, ax, main_color=colors[1], second_color=colors[0]):
    
    sns.regplot(data=df, x=x, y=y, ax=ax, 
                 color=main_color, ci=75,
                scatter_kws={
                    'edgecolor':second_color,
                    'linewidth':1.5,
                    's':50
                },
                line_kws={
                    'color':colors[2],
                    'linewidth':3,
                }
               )
    ax.set_xlabel(x.replace("_", " ").capitalize())
    ax.set_ylabel(y.replace("_", " ").capitalize())
    
    sns.despine(ax=ax)
    ax.grid(axis='x')

cols = ['Coe_left(yrs)', 'Dep($)', 'Mileage(mile)', 'Road Tax($)', 'Dereg Value($)', 'COE($)', 'Engine Cap(cc)',
       'Curb Weight(kg)', 'Manufactured(yrs)', 'OMV($)', 'ARF($)', 'Power', 'No. of Owners' ]

fig, axs = plt.subplots(8, 2,figsize=(15, 40))

for i, col in enumerate(cols):
    
    row_index = i // 2
    col_index = i % 2
    
    ax = axs[row_index][col_index]
    
    scatter(df, col,'Price($)', ax)
    
plt.show()
```

![graph3](https://user-images.githubusercontent.com/85028821/189383800-71acc109-ea9d-45d7-a8d8-ab4374afeb5a.png)

ข้อสังเกตุที่เราได้จากกราฟต่างๆจะเห็นว่า Dep, Coe left,Road tax, Dereg Value, Coe,Engine cap, curb weight, Omv, Art, Power มีความสัมพันธ์เชิงบวกกับราคารถยนต์มือ2ในสิงคโปร์ส่วน Mileage, Manufactured, No. Of owners มีความสัมพันธ์เชิงลบกับราคารถ ก็ทำให้เกิดข้อสงสัยว่ามีความ Strong correlation มากเพียงใดในแต่ละปัจจัย เลยใช้ heatmap ในการอธิบายต่อไป

```
def corr_map(df, ax, palette, edgecolor=colors[0]):
    
    corr = df.corr()
    
    sns.heatmap(corr, annot=True, ax=ax,
               cmap=palette, square=True, linewidth=.5, linecolor=edgecolor,
               vmin=-1, vmax=1, fmt=".2f")
 
 fig, ax = plt.subplots(figsize=(12, 12))

palette = sns.diverging_palette(299, 192, s=89, l=71, as_cmap=True, sep=20)

corr_map(df, ax, palette)
```

![corr_matrix](https://user-images.githubusercontent.com/85028821/189384778-fc9d8017-2fae-454c-b937-95d3a38f4b7b.png)

จาก heatmap ข้างต้นปัจจัยที่มีความ Strong correlation มากที่สุด 5 อันดับแรก ได้แก่ 	Dereg Value, ARF, OMV , Power และ Dep
ซึ่งอาจจะเป็นปัจจัยที่สำคัญหลักๆที่มีผลต่อราคารถยนต์มือ2ในสิงคโปร์

# Data preprocessing

## Create Dummy Variable
เป็นการเปลี่ยน column ที่เป็นตัวแปรประเภท category สามารถเข้าสมการ regression ได้ โดยจะได้ค่าออกมาเป็น column ใหม่ ที่เป็นข้อมูลประเภท binary ความหมายคือถ้าเป็นข้อมูลประเภทนั้นจะแทนด้วย 1 ถ้าไม่ใช่แทนด้วย 0

```
df1 = pd.get_dummies(data=df, drop_first=True)
```
ตัวอย่างตารางหลังการทำ dummy variable

![image](https://user-images.githubusercontent.com/85028821/189478547-b44aec4e-b30e-412b-bfd8-5bbcd15c5cfc.png)

## Data spliting
ทำการ split data ออกเป็น train และ test ทั้งตัวแปล x และ y โดยกำหนดอัตราส่วนเป็น 70:30 ตามลำดับ และกำหนด random_state=15

```
X = df1.drop(['Price($)'], axis=1).values
y = df1['Price($)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15, test_size=0.30)
```

## Data scaling
เป็นการปรับข้อมูลให้มี scale อยู่ในระดับเดียวกัน เพื่อนำไปเข้าสมการ regresssion ได้มีประสิทธิภาพมากขึ้น โดยวิธีการนี้จะทำให้ข้อมูลในแต่ละ column ถูกปรับขนาดให้มีความแปรปรวนใกล้เคียงกันเท่ากับ 1
```
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE

ss = StandardScaler()

ss.fit(X)

X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)
```

# Traditional Machine Learning

## Model 1 : Polynomial regression + Ridge

```
from datetime import datetime
start_time = datetime.now()

from sklearn.linear_model import Ridge

poly_reg=PolynomialFeatures(degree=3)

X_poly = poly_reg.fit_transform(X_train_scaled)

X_poly_test = poly_reg.fit_transform(X_test_scaled)

train_mae = list()
test_mae = list()

alphas = [i for i in range(0, 2000, 100)]
for i in alphas :
    clf = Ridge(alpha=i)
    clf.fit(X_poly, y_train)
    
    y_pred = clf.predict(X_poly)
    y_pred_test = clf.predict(X_poly_test)
        
    train_mae.append(mean_absolute_error(y_train, y_pred))
    test_mae.append(mean_absolute_error(y_test, y_pred_test))
    
for i, alpha in enumerate(alphas):
    print("--"*10, f" alpha={alpha} ", "--"*10)
    print(f"TRAIN MAE -> {train_mae[i]}")
    print(f"TEST MAE -> {test_mae[i]}")

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ทำ Traditional Machine learning modelที่1 ด้วย model Polynomial Regression degree3 และใช้ Ridge ในการทำ regularization
ทำการปรับค่า alpha ของ ridge เพื่อหา loss ที่ต่ำที่สุดของ test ที่ไม่ทำให้เกิดการ overfitting โดย loss function ที่เลือกใช้คือ Mean absolute error 

```
i_alpha_optim = np.argmin(test_mae)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)
```
โดยค่า alpha ที่ทำให้ค่า loss ต่ำที่สุดจากการทดลองครั้งนี้คือ alpha = 800 ได้ค่า MAE ของ test เท่ากับ 7690.42 และของ train เท่ากับ 3026.06
หรือดูจากกราฟระหว่าง MAE และ alpha ได้เช่นกัน
```
plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_mae, label="Train")
plt.semilogx(alphas, test_mae, label="Test")
plt.xlim([0, 2000])
plt.ylim([0, 10000])
plt.xlabel('alpha')
plt.ylabel('MAE')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/189691895-1aea8bfa-9146-407a-8f41-524d1d14533e.png)

## Model 2 : DecisionTreeRegressor
```
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
regr_1 = DecisionTreeRegressor(max_depth=60,min_samples_split=6,min_weight_fraction_leaf=0.001,max_features="auto")
regr_1.fit(X_train_scaled,y_train)
y_pred_train = regr_1.predict(X_train_scaled)
y_pred_test = regr_1.predict(X_test_scaled)
print('MAE:', metrics.mean_absolute_error(y_train, y_pred_train))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_test))
```
ทำ Traditional Machine learning modelที่2 ด้วย model DecisionTreeRegressor โดยปรับค่า hyperparameter ดังนี้
- max_depth=60
- min_samples_split=6
- min_weight_fraction_leaf=0.001
- max_features="auto"

# Multilayer perceptron (MLP)

## Data formatting

```
y_test = y_test.reshape(-1,1)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
```
ทำการเปลี่ยน type ของข้อมูลให้เป็น float32 และปรับshapeให้แน่ใจเพื่อเข้าโมเดล MLP

## Create Network architecture

```
np.random.seed(1234)
tf.random.set_seed(5678)

model = tf.keras.models.Sequential()

# Input layer
model.add( tf.keras.Input(shape=(61, ) ))
# Dense
model.add(tf.keras.layers.Dense(64, activation='relu', name = 'dense1'))
model.add(tf.keras.layers.Dense(32, activation='relu', name = 'dense2'))
model.add(tf.keras.layers.Dense(16, activation='relu', name = 'dense3'))
model.add(tf.keras.layers.Dense(8, activation='relu', name = 'dense4'))
# Output layer
model.add(tf.keras.layers.Dense(1, activation='linear', name = 'output') )
```
จากการทดลองเรามีการเลือกใช้จำนวน node มากๆ แต่ได้ค่า error ออกมามากกว่า model ปัจจุบันที่ใช้จำนวน node น้อยกว่า
ดังนั้น โมเดลของเราจึงมีการปรับค่าต่างๆดังนี้

- set seed : กำหนดค่า seed ไว้ที่ 1234 และ 5678
- input layer : กำหนดจำนวน node ของinput เท่ากับจำนวน column คือ 61 
- dense layer : สร้าง dense layer 4 layer กำหนดจำนวน node ของแต่ละ layer เป็น 64,32,16,8 ตามลำดับ ใช้activation function คือ relu
- output layer : กำหนดจำนวน node ของoutput เท่ากับ 1 ใช้activation function คือ linear

โดย model นี้ไม่ได้ทำการ regularization เนื่องจาก model ไม่ได้มีปัญหา overfitting

Reference : https://github.com/christianversloot/machine-learning-articles/blob/main/creating-an-mlp-for-regression-with-keras.md

## Network diagram

![MicrosoftTeams-image (7)](https://user-images.githubusercontent.com/85028821/189722913-80b6351a-dd45-48ee-861e-887f37436de8.png)

https://app.diagrams.net/

## Compile the model
```
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
```
ใช้ loss function และ matrics เป็น mean_absolute_error, ใช้ optimizer เป็น adam


## Trained model with seed
```
from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)

history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=2, verbose=1, validation_split=0.3, callbacks=[earlystopping])
```
ทำการtrain และปรับ hyperparameter ดังนี้
- epochs=200 รอบ 
- batch size = 2 
- validation_split = 0.3
- callbacks ที่ใช้คือ EarlyStopping โดยทำการเลือกค่า weight ที่ได้ค่า val_loss ต่ำที่สุด และถ้าไม่ได้ค่า val_loss ที่ต่ำกว่าเดิมจะมีการrunไปอีก20 epoch แล้วหยุดการทำงาน 

GPU ที่ใช้คือ Tesla T4 (UUID: GPU-7988356d-dfd4-b48c-471f-8ec9597324b4)
เวลาที่ใช้ในการtrain model 1 ตัวคือ 3นาที 22วินาที

กราฟของ loss และ epoch
```
# Summarize history for loss
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.grid()
plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/189704163-9c6eb463-7952-4dec-8f0c-8581cb5abaae.png)

ผลการทดลองที่ได้จาก model นี้ คือได้ค่า loss: 5507.1641 และ val_loss: 7592.5439 เป็นค่า val_loss ที่ต่ำที่สุดใน epoch ที่ 77

## Evaluation the model on test set
ทำการ train model แต่ละรอบโดยการเอาค่า seed ออก และนำ test dataset มาทำนายค่า output
และการประเมินประสิทธิภาพของ model จากค่า MAE

```
model = tf.keras.models.Sequential()

# Input layer
model.add( tf.keras.Input(shape=(61, ) ))
# Dense
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
# Output layer
model.add(tf.keras.layers.Dense(1, activation='linear', name = 'output') )

# Configure the model and start training
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=2, verbose=1, validation_split=0.3, callbacks =[earlystopping])

y_pred_test = model.predict( X_test_scaled )
print(mean_absolute_error(y_test,y_pred_test))
```
- รอบที่1 ใช้เวลา 2min43sec :  MAE of test dataset = 7367.43
- รอบที่2 ใช้เวลา 3min06sec :  MAE of test dataset = 7849.03
- รอบที่3 ใช้เวลา 3min26sec :  MAE of test dataset = 7149.07
- รอบที่4 ใช้เวลา 3min18sec :  MAE of test dataset = 7373.33
- รอบที่5 ใช้เวลา 4min15sec :  MAE of test dataset = 7112.05

ได้ค่าเฉลี่ยของ MAE ออกมาเท่ากับ 7370.18 และ SD เท่ากับ 262.61

![image](https://user-images.githubusercontent.com/85028821/189704163-9c6eb463-7952-4dec-8f0c-8581cb5abaae.png)

จากกราฟเทียบ train vs. validation สรุปได้ว่าข้อมูลของเราไม่ได้เกิดปัญหา underfit หรือ overfit เนื่องจากเราเลือกใช้ epoch ที่ทำให้ค่า error ของทั้ง train และ validation ต่ำที่สุดในmodelนี้

Evaluation metric ที่ใช้คือ MAE เนื่องจากข้อมูลที่เลือกมาใช้มีลักษณะเป็น regression ดังนั้นจึงใช้ MAE เป็นทั้ง loss function และ evaluation metric

# Conclusion
จากการทำ Traditional ML ทางกลุ่มเราเลือก algorithm มา2แบบ ได้แก่ Polynomial regression + Ridge และ DecisionTreeRegressor โดยใช้การประเมินmodel จากค่า mean absolute error (MAE) 
และเวลาที่ใช้ในการrun ส่วนทาง Multilayer perceptron ใช้การประเมินแบบเดียวกัน ดังตาราง

![image](https://user-images.githubusercontent.com/85028821/189803151-8bf9bedc-abf5-44e1-8804-f3bca7453f76.png)


ผลลัพท์ที่ได้จากค่า MAE ของทั้ง Traditional ML และ MLP ไม่ได้แตกต่างกันมาก อาจเนื่องจากว่าข้อมูลที่นำมาใช้ไม่ได้ซับซ้อน หรือ ปริมาณไม่มาก และเวลาการrun ของ MLP ใช้เวลาค่อนข้างนานกว่า Traditional ML ดังนั้น จากข้อมูลที่เรามีซึ่งอาจไม่มีปริมาณมากพอที่เหมาะสมกับการทำ MLP (Regression) เนื่องจากใช้เวลาค่อนข้างมากกว่า Traditional ML ถึง 2 เท่าแต่กลับให้ผลลัพท์ที่พอๆกัน แต่ถ้ามีเวลาในการทดลองที่มากขึ้นอาจสามารถหาโมเดล MLP ที่ได้ค่า error ที่เหมาะสมกว่านี้ได้เนื่องจากโมเดล MLP มีการปรับค่าได้หลากหลายและซับซ้อนมากกว่า

# Member 
- นส.ศิริวลัย   มณีสินธุ์    6410412011
- นายศิวกร    ศรีชัยพฤกษ์ 6410412012
- นายนนทพร  วงษ์เล็ก    6410412016
- นายธีรพล   แสงเมือง    6410412019

