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



# Heading 1
## Heading 2
```
print(hello world)
```
## Heading 2
### Heading 3

![download](https://user-images.githubusercontent.com/85028821/189327354-413612c2-edf3-4ab8-9d20-0927cc484351.png)
