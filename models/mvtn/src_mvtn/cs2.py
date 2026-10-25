import pandas as pd
import numpy as np

o = pd.read_csv('csv/NVGestures/original.csv', header = None) 

df1 = pd.read_csv('csv/NVGestures/train_nv_rgb.csv', header = None) # place your csv1 in df1
df2 = pd.read_csv('csv/NVGestures/train_nv_normal.csv', header = None) # place your csv2 in df2
print((df1).shape)
print((df2).shape)

o1 = o.iloc[:,:].values.tolist() 
#print(type(o1))

rate_in_1 = df1.iloc[:,:].values.tolist() #store the values of the 3rd column from csv1 to a list
rate_out_1 = df2.iloc[:,:].values.tolist() #store the values of the 4th column from csv1 to a list

#rate_in_2 = df2.iloc[:,2].values.tolist() #store the values of the 3rd column from csv1 to a list
#rate_out_2 = df2.iloc[:,3].values.tolist() #store the values of the 4th column from csv1 to a list

rate_in_total = [max(x,y) for (x, y) in zip(rate_in_1, rate_out_1)] # add the values of 2 rate in lists into rate_in_total list
# rate_in_total = [x+y for x, y in zip(rate_out_1, rate_in_1)] # add the values of 2 rate out lists into rate_out_total list

#print(rate_in_total[1])
#Now to output/concatenate this into 1 DataFrame:

final_df = pd.DataFrame(rate_in_total)
#final_df['Node'] = ['allnode' for x in rate_in_total]
#final_df['Link'] = df1.iloc[:,1].values.tolist()
#final_df = rate_in_total
#final_df['rate-out'] = rate_out_total
with open('csv/NVGestures/rgb_normal.csv', 'a', newline='') as csvfile:
	final_df.to_csv(csvfile, mode='a',header=False)
#print(np.where(max(rate_in_total[1])))

#print(len(rate_in_total))
c=0
for x in range(len(rate_in_total)):
	#print(type(rate_in_total[x]))
	#print(max(rate_in_total[x]))
	if np.argmax(rate_in_total[x], axis=0)==o1[x]:
		c +=1
print( c / 482)

##swith open('csv/Briareo/original.csv', 'a', newline='') as csvfile:
