from pickle import TRUE
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import sys
from matplotlib import pyplot as plot

file_path = 'weather.xlsx'

perceptions = []
tempMAXS = []
tempMINS = []
rains = []

cont_names = []
cont_count = []
cont_missing = []
cont_uniques = []
cont_mins = []
cont_maxs = []
cont_quartiles_1 = []
cont_quartiles_3 = []
cont_averages = []
cont_medians = []
cont_stdevs = []

def readFile(path):
    global perceptions
    global tempMAXS
    global tempMINS
    global rains

    data = pd.read_excel(path)
    df = pd.DataFrame(data)
    perceptions = df['PRCP'].to_list()
    tempMAXS = df['TMAX'].to_list()    
    tempMINS = df['TMIN'].to_list()    
    rains = df['RAIN'].to_list()    
    rains = [1 if val == 'True' or val == "TRUE" or val == True else 0 for val in rains]

# Suranda trukstamas reiksmes
def find_missing_values(column):
    indices = []

    for i in range(len(column)):
        if column[i] == ' ' or column[i] == '' or column[i] == 'NA':
            indices.append(i)
    return indices

# Atrenka nesikartojancias reiksmes
def unique_values(column):
    temp = []

    for value in column:
        if not temp.__contains__(value):
            temp.append(value)
    return temp

# Suranda pirma ir antra moda
# maxVal - pirma moda, maxVal2 - antra moda
def moda(uniques, column):
    temp = []

    for unique in uniques:
        count = 0
        for value in column:
            if unique == value:
                count+=1
        temp.append({'value': unique, 'count' : count})

    maxVal = temp[0]
    maxVal2 = None

    for value in temp:
        if value['count'] > maxVal['count']:
            maxVal2 = maxVal
            maxVal = value
        elif maxVal['value'] != value['value'] and (maxVal2 is None or value['count'] > maxVal2['count']):
            maxVal2 = value

    return maxVal, maxVal2

# Suranda mediana
def median(column):
    print(len(column))
    index = (int)(len(column)/2)
    copy = column.copy()
    copy.sort()

    if len(copy)%2 == 1:
        return copy[index]
    else:
        return (copy[index]+copy[index-1])/2

# Suranda maziausia ir didziausia reiksme
def min_and_max(column):
    min = sys.float_info.max
    max = sys.float_info.min

    for value in column:
        if value > max:
            max = value
        if value < min:
            min = value
    return {'min': min, 'max': max}

# Suskaiciuoja vidurki
def average(column):
    sum=0

    for value in column:
        sum+=value
    return sum/len(column)

# Suskaiciuoja dispersija
def variance(avg, column):
    sum=0

    for val in column:
        sum+=np.square(val-avg)
    return sum/(len(column)-1)

# Grazina standartini nuokrypi
def std_dev(var):
    return np.sqrt(var)

# Grazina kvartili pagal parametra precentile pvz: precentile = 25% - pirmasis kvartilis
def quartile(column, precentile):
    copy = column.copy()
    copy.sort()
    index = (int)(len(copy)*precentile/100)
    return copy[index]

# Atlieka visus veiksmus su vienu tolydziuoju duomenu stulpeliu
def continous(column):
    count = len(column)
    missing_values = find_missing_values(column)
    uniques = unique_values(column)
    _median = median(column)
    min_max = min_and_max(column)
    avg = average(column)
    var = variance(avg, column)
    stdev = std_dev(var)
    quartile_1 = quartile(column, 25)
    quartile_3 = quartile(column, 75)
    missing_pct = len(missing_values)/count * 100
    return count, missing_pct, uniques, _median, min_max, avg, stdev, quartile_1, quartile_3

# Uzpildo tolydziuju duomenu stulpeliu masyvus, kurie reikalingi isvesti duomenis apie kiekviena duomenu stulpeli i excel faila
def fill_continous(column, name, names, counts, missings, _uniques, mins, maxs, quartiles_1, quartiles_3, averages, medians, std_devs):
    count, missing_pct, uniques, _median, min_max, avg, stdev, quartile_1, quartile_3 = continous(column)
    names.append(name)
    counts.append(count)
    missings.append(missing_pct)
    _uniques.append(len(uniques))
    mins.append(min_max['min'])
    maxs.append(min_max['max'])
    quartiles_1.append(quartile_1)
    quartiles_3.append(quartile_3)
    averages.append(avg)
    medians.append(_median)
    std_devs.append(stdev)


# Apskaiciuoja kovariacija
def cov(acol, bcol, aavg, bavg):
    n = len(acol)
    sum=0

    for i in range(n):
        sum+=(acol[i] - aavg) * (bcol[i]-bavg)
    return sum/(n-1)

# Apskaiciuoja koreliacija
def corr(cov, std_a, std_b):
    return cov/(std_a*std_b)

# Normalizuoja duomenu reiksmes
def norm(column, High, Low):
    _min = min(column)
    _max = max(column)
    return ((np.array(column) - _min)/(_max - _min)) * (High - Low) + Low

if __name__ == '__main__':
    readFile(file_path)

    fill_continous(perceptions, 'PRCP', cont_names, cont_count, cont_missing, cont_uniques, cont_mins, cont_maxs, cont_quartiles_1, cont_quartiles_3,
    cont_averages, cont_medians, cont_stdevs)

    fill_continous(tempMAXS, 'TMAX', cont_names, cont_count, cont_missing, cont_uniques, cont_mins, cont_maxs, cont_quartiles_1, cont_quartiles_3,
    cont_averages, cont_medians, cont_stdevs)

    fill_continous(tempMINS, 'TMIN', cont_names, cont_count, cont_missing, cont_uniques, cont_mins, cont_maxs, cont_quartiles_1, cont_quartiles_3,
    cont_averages, cont_medians, cont_stdevs)

    fill_continous(rains, 'RAIN', cont_names, cont_count, cont_missing, cont_uniques, cont_mins, cont_maxs, cont_quartiles_1, cont_quartiles_3,
    cont_averages, cont_medians, cont_stdevs)

    excel_cont_data = {'Atributo pavadinimas': cont_names,'Kiekis' : cont_count,'Trūkstamos reikšmės, %': cont_missing, 'Kardinalumas' : cont_uniques, 
            'Minimali reikšmė': cont_mins, 'Maksimali reikšmė': cont_maxs, '1-asis kvartilis': cont_quartiles_1, '3-iasis kvartilis': cont_quartiles_3,
            'Vidurkis': cont_averages, 'Mediana': cont_medians, 'Standartinis nuokrypis': cont_stdevs}

    df = pd.DataFrame(excel_cont_data, columns = ['Atributo pavadinimas','Kiekis','Trūkstamos reikšmės, %', 'Kardinalumas', 
            'Minimali reikšmė', 'Maksimali reikšmė', '1-asis kvartilis', '3-iasis kvartilis',
            'Vidurkis', 'Mediana', 'Standartinis nuokrypis'])
    df.to_excel('tolydiniai.xlsx', index=False)


   