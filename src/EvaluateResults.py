import torch
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




################################################################################
### Loading Data 
################################################################################
show=True
metadata = pd.read_csv("metadata.csv", skipinitialspace=True)
metadata['file']

movies = {}
path = './analysis/james-bond/'
for file in os.listdir(path): 
    fpath = os.path.join(path,file)
    with open(fpath, 'rb') as f: 
        loaded = pickle.load(f)
        movies[file] = loaded



mean = []
for (file, movie) in movies.items():
    t = movie['traits']
    mask_men   = t[:,0] > 0
    means = torch.stack([(-t[mask_men,:].mean(dim=0) +1)/2, 
                         (-t[~mask_men,:].mean(dim=0)+1)/2])
    mean.append(means)
mean = torch.stack(mean)



metadata['men_sexualization']   = mean[:,0,2]
metadata['women_sexualization'] = mean[:,1,2]
metadata['men_helplessness']   = mean[:,0,1]
metadata['women_helplessness'] = mean[:,1,1]
sex_m = metadata['men_sexualization']  
sex_f = metadata['women_sexualization']
ag_m  = metadata['men_helplessness']  
ag_f  = metadata['women_helplessness']
metadata['title_year'] = metadata['Film'] + ': ' + metadata['Year'].astype(str) 


################################################################################
### Line Plot Sexualization
################################################################################

years = metadata['Year']
plt.plot(years, sex_m, label='men')
plt.plot(years, sex_f, label='women')
plt.ylim([0,1])
plt.ylabel("Degree of sexualization")
plt.xlabel("Release Date")
plt.title("Sexualization by film and gender")
plt.legend()
plt.savefig("report/assets/sexualization.png")
if show:
    plt.show()
plt.close()


################################################################################
### Line Plot Helplessness
################################################################################
plt.plot(years, ag_m, label='men')
plt.plot(years, ag_f, label='women')
plt.ylim([0,1])
plt.ylabel("Degree of helplessness")
plt.xlabel("Release Date")
plt.title("Helplessness by film and gender")
plt.legend()
plt.savefig("report/assets/helplessness.png")
if show:
    plt.show()
plt.close()






################################################################################
### Scatter Plot Helplessness
################################################################################
plt.scatter(years, ag_m, label='men')
plt.scatter(years, ag_f, label='women')
z_m = np.polyfit(years, ag_m, 1)
p_m = np.poly1d(z_m)
plt.plot(years, p_m(years))
z_f = np.polyfit(years, ag_f, 1)
p_f = np.poly1d(z_f)
plt.plot(years, p_f(years))
plt.ylim([0,1])
plt.ylabel("Degree of helplessness")
plt.xlabel("Release Date")
plt.title("Helplessness by film and gender")
plt.legend()
plt.savefig("report/assets/scatter-helplessness.png")
if show:
    plt.show()
plt.close()


################################################################################
### Scatter Plot Sexualization
################################################################################
years = metadata['Year']
plt.scatter(years, sex_m, label='men')
plt.scatter(years, sex_f, label='women')
z_m = np.polyfit(years, sex_m, 1)
p_m = np.poly1d(z_m)
plt.plot(years, p_m(years))
z_f = np.polyfit(years, sex_f, 1)
p_f = np.poly1d(z_f)
plt.plot(years, p_f(years))
plt.ylim([0,1])
plt.ylabel("Degree of sexualization")
plt.xlabel("Release Date")
plt.title("Sexualization by film and gender")
plt.legend()
plt.savefig("report/assets/scatter-sexualization.png")
if show:
    plt.show()
plt.close()


################################################################################
### Bar Plot Sex Film 
################################################################################
titles = metadata['title_year']
x = np.arange(len(titles))  
width = 0.35  

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, sex_m, width, label='men')
bars2 = ax.bar(x + width/2, sex_f, width, label='women')

# Add labels and legend
ax.set_ylabel("Degree of sexualization")
ax.set_ylim([0,1])
ax.set_title('Sexualization by film and gender')
ax.set_xticks(x)
ax.set_xlabel('Movie Title (Release)')
ax.set_xticklabels(titles, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("report/assets/sexualization-title.png")
if show:
    plt.show()
plt.close()






################################################################################
### Bar Plot Sex Actor
################################################################################
data = (metadata.groupby('Actor')
        .agg({'women_sexualization': 'mean', 'men_sexualization': 'mean'})
        .sort_values(by='women_sexualization', ascending=False))


titles = data.index
x = np.arange(len(titles))  
width = 0.35  

fig, ax = plt.subplots()
bars1 = ax.bar(x + width/2, data['men_sexualization'], width, label='men')
bars2 = ax.bar(x - width/2, data['women_sexualization'], width, label='women')

# Add labels and legend
ax.set_ylabel('Values')
ax.set_ylabel("Degree of sexualization")
ax.set_title('Mean Sexualization by James Bond Actor')
ax.set_xticks(x)
ax.set_xlabel('James Bond Actor')
ax.set_xticklabels(titles, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.savefig("report/assets/sexualization-actor.png")
if show:
    plt.show()
plt.close()



################################################################################
### Bar Plot Sex Director
################################################################################
data = (metadata.groupby('Director')
        .agg({'women_sexualization': 'mean', 'men_sexualization': 'mean'})
        .sort_values(by='women_sexualization', ascending=False))


titles = data.index
x = np.arange(len(titles))  
width = 0.35  

fig, ax = plt.subplots()
bars1 = ax.bar(x + width/2, data['men_sexualization'], width, label='men')
bars2 = ax.bar(x - width/2, data['women_sexualization'], width, label='women')

# Add labels and legend
ax.set_ylabel('Degree of sexualization')
ax.set_title('Mean Sexualization by Director')
ax.set_xticks(x)
ax.set_xlabel('Director')
ax.set_xticklabels(titles, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.savefig("report/assets/sexualization-director.png")
if show:
    plt.show()
plt.close()


################################################################################
### Whisker Plot by Movie
################################################################################

# t = movies['1962-dr-no.pkl']['traits']
# mask_men   = t[:,0] > 0
# t = (1-t)/2
# data = [t[mask_men,2], t[~mask_men,2]]
#
# plt.boxplot(data)
# plt.xticks([1,2], ['men', 'women'])
# plt.show()




################################################################################
### Testing
################################################################################
print("women trend: 1962={} and 2008={}".format(p_f(1962), p_f(2008)))
print("men trend: 1962={} and 2008={}".format(p_m(1962), p_m(2008)))
