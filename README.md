# Introduction
จุดประสงค์การศึกษาครั้งนี้ต้องการเปรียบเทียบประสิทธิภาพการทำงานระหว่าง traditional machine learning กับ MLP

# Data
## Data source 
data เกี่ยวกับอะไร -> ข้อมูลที่คาดว่าจะส่งผลต่อราคารถมือ2ในสิงคโปร
data columns -
             -
             -
             -
```
df.info()
```

## Cleansing data
อธิบายว่าทำอะไร เอา na ออก ลบ column

```
df = df.replace('N.A', np.NaN)
df = df.replace('N.A.', np.NaN)
del df['Unnamed: 18']
df = df.dropna()
```
นำข้อมูลเข้า openrefine - เปลี่ยน type ข้อมูล / เปลี่ยน brand รถ

# EDA
ดูการกระจายตัวของข้อมูล
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

ได้อะไรจากกราฟ

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

จากกราฟข้างต้นจะเห็นได้ว่า
รถที่นิยม 5 อันดับแรกคือ
ชนิดรถที่นิยม ...
ชนิดเกียร์ ...

```
df['Brand'].value_counts().head()

df['Type'].value_counts().head()

df['Transmission'].value_counts().head()
```

แปะตาราง

## Correlation
หาความสัมพันธ์ของตัวแปรต่างๆว่ามีผลต่อราคารถยนต์อย่างไร
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

อธิบายว่าปัจจัยไหนมีผลต่อราคายังไง

```
def stripplot(df, x, y, ax, palette=[colors[1], colors[2]]):
    
    sns.stripplot(data=df, x=x, y=y, palette=palette, ax=ax,
                 linewidth=2, size=8)
    
    set_style(ax)

fig, axs = plt.subplots(1, 3, figsize=(25, 6), sharey=True)

for i, col in enumerate(['Brand', 'Type', 'Transmission']):
    
    stripplot(df, col, 'Price($)', axs[i])
    
    axs[i].set(
        xlabel=col.replace("_", " ").capitalize(),
        ylabel="Price($)"
    )

plt.show()
```
![graph4](https://user-images.githubusercontent.com/85028821/189384404-9842b557-9985-4036-a873-5a85643ef653.png)

อธิบายกราฟ

เพื่อดูค่าความสัมพันธ์ที่ชัดเจน สามารถดูจากตาราง correlation

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

เรียงลำดับ corr ของปัจจัยที่ส่งผลต่อ ราคา

# Data preprocessing

## Create Dummy Variable
อธิบายว่าทำทำไม
```
df1 = pd.get_dummies(data=df, drop_first=True)
```

![image](https://user-images.githubusercontent.com/85028821/189478547-b44aec4e-b30e-412b-bfd8-5bbcd15c5cfc.png)

## Data spliting
อธิบาย
```
X = df1.drop(['Price($)'], axis=1).values
y = df1['Price($)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15, test_size=0.30)
```
## Data scaling
อธิบาย เพื่อ
```
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE

ss = StandardScaler()

ss.fit(X)

X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)
```
# Traditional Machine Learning
ใช้วิธีอะไร ปรับค่าอะไรยังไงบ้าง ได้ MAE เป็นยังไง

# Multilayer perceptron
- Network architecture: รายละเอียดต่าง ๆ ของโมเดลที่เลือกใช้ (เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) ในรูปแบบของ network diagram หรือตาราง (โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่าน จะสามารถไปสร้าง network ตามเราได้)
- Training: รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size,
epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ
- Results: แสดงตัวเลขผลลัพธ์ในรูปของค่าเฉลี่ย mean±SD โดยให้ทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกันอย่างน้อย 3-5 รอบเพื่อให้ได้อย่างน้อย 3-5 โมเดลมาหาประสิทธิภาพเฉลี่ยกัน, แสดงผลลัพธ์การ train โมเดลเป็นกราฟเทียบ train vs. validation, สรุปผลว่าเกิด underfit หรือ overfit หรือไม่, อธิบาย evaluation metric ที่ใช้ในการประเมินประสิทธิภาพของโมเดลบน train/val/test sets ตามความเหมาะสมของปัญหา, หากสามารถเปรียบเทียบผลลัพธ์ของโมเดลเรากับโมเดลอื่น ๆ (ของคนอื่น) บน any standard benchmark dataset ได้ด้วยจะยิ่งทำให้งานดูน่าเชื่อถือยิ่งขึ้น เช่น เทียบความแม่นยำ เทียบเวลาที่ใช้train เทียบเวลาที่ใช้ inference บนซีพียูและจีพียู เทียบขนาดโมเดล ฯลฯ

# Conclusion
88888888888888888888888888
eieiei
