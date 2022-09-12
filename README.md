# Introduction
จุดประสงค์การศึกษาครั้งนี้ต้องการเปรียบเทียบประสิทธิภาพการทำงานระหว่าง traditional machine learning กับ MLP โดยใช้ชุดข้อมูลตัวอย่างเป็น ราคารถยนต์มือสองในสิงคโปร์ปี2021 จาก https://www.kaggle.com/datasets/jiantay33/singapore-used-car

# Data
## Data source 
ที่สิงคโปร์ ตลาดรถมือสองมีปัจจัยหลายอย่างที่ส่งผลกับหลายค่า เนื่องจากการมีการควบคุมกฎหมาย ในcolumnต่างๆคือปัจจัยที่ส่งผลต่อการเลือกซื้อรถยนต์ 1 คัน ในสิงคโปร์

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

```
df.info()
```

## Cleansing data

เราทำการ clean ข้อมูลผ่าน application ชื่อ Openrefine โดยการกรุ๊ปชื่อตัวแปรให้เข้าใจง่าย เช่น Brand ของรถยนต์, เปลี่ยน type ของตัวแปร
ให้เป็น numerical ยิ่งกว่านั้นเราใช้ python ทำการกำจัด null และ coloumn ที่ไม่จำเป็นออก

```
df = df.replace('N.A', np.NaN)
df = df.replace('N.A.', np.NaN)
del df['Unnamed: 18']
df = df.dropna()
```

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
ต่อมาเราจะดูปัจจัยต่างๆที่มีผลต่อราคารถโดยสังเกตุจากความสัมพันธ์ระหว่างตัวแปรต่างๆข้างต้นกับราคารถเราจะได้กราฟความสัมพันธ์ดังนี้
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

ข้อสังเกตุที่เราได้จากกราฟต่างๆจะเห็นว่า Dep, Coe left,Road tax, Dereg Value, Coe,Engine cap, curb weight, Omv, Art, Power มีความสัมพันธ์เชิงบวกกับราคารถยนต์ส่วน Mileage, Manufactured, No. Of owners มีความสัมพันธ์เชิงลบกับราคารถ ก็ทำให้เกิดข้อสงสัยว่ามีความ Strong correlation มากเพียงใดในแต่ละปัจจัย เลยใช้ heatmap ในการอธิบายต่อไป

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

จาก heat map ข้างต้นปัจจัยที่มีความ Strong correlation มากที่สุด 5 อันดับแรก ได้แก่ 	Dereg Value, ARF, OMV , Power และ Dep
ซึ่งอาจจะเป็นปัจจัยที่สำคัญหลักๆที่มีผลต่อราคา

# Data preprocessing

## Create Dummy Variable
เป็นการเปลี่ยน column ที่เป็นตัวแปรประเภท category สามารถเข้าสมการ regression ได้ โดยจะได้ค่าออกมาเป็น column ใหม่ ที่เป็นข้อมูลประเภท binary ความหมานคือถ้าเป็นข้อมูลประเภทนั้นจะแทนด้วย 1 ถ้าไม่ใช่แทนด้วย 0
```
df1 = pd.get_dummies(data=df, drop_first=True)
```
ตัวอย่างตารางหลังการทำ dummy variable เรียบร้อย

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
ใช้วิธีอะไร ปรับค่าอะไรยังไงบ้าง ได้ MAE เป็นยังไง

# Multilayer perceptron
- Network architecture: รายละเอียดต่าง ๆ ของโมเดลที่เลือกใช้ (เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) ในรูปแบบของ network diagram หรือตาราง (โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่าน จะสามารถไปสร้าง network ตามเราได้)
- Training: รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size,
epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ
- Results: แสดงตัวเลขผลลัพธ์ในรูปของค่าเฉลี่ย mean±SD โดยให้ทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกันอย่างน้อย 3-5 รอบเพื่อให้ได้อย่างน้อย 3-5 โมเดลมาหาประสิทธิภาพเฉลี่ยกัน, แสดงผลลัพธ์การ train โมเดลเป็นกราฟเทียบ train vs. validation, สรุปผลว่าเกิด underfit หรือ overfit หรือไม่, อธิบาย evaluation metric ที่ใช้ในการประเมินประสิทธิภาพของโมเดลบน train/val/test sets ตามความเหมาะสมของปัญหา, หากสามารถเปรียบเทียบผลลัพธ์ของโมเดลเรากับโมเดลอื่น ๆ (ของคนอื่น) บน any standard benchmark dataset ได้ด้วยจะยิ่งทำให้งานดูน่าเชื่อถือยิ่งขึ้น เช่น เทียบความแม่นยำ เทียบเวลาที่ใช้train เทียบเวลาที่ใช้ inference บนซีพียูและจีพียู เทียบขนาดโมเดล ฯลฯ

# Conclusion
88888888888888888888888888
eieiei
