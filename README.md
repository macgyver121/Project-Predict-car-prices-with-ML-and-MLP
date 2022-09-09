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

## Cleasing data
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



# Heading 1
## Heading 2
```
print(hello world)
```
## Heading 2
### Heading 3

![download](https://user-images.githubusercontent.com/85028821/189327354-413612c2-edf3-4ab8-9d20-0927cc484351.png)
