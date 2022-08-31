
# Pandas

`Pandas` to kolejna biblioteka niezbędna do analizy danych w Pythonie. Dostarcza wydajne struktury danych, dzięki którym praca z danymi tabularycznymi staje się prosta i ituicyjna. Celem twórców jest utrzymanie statusu biblioteki niezbędnej do codziennych analiz oraz zdobycie fotela lidera w kategorii najpotężniejszego narzędzia open-source do analizy danych w jakimkolwiek języku programowania. Obecnie, projekt wciąż prężnie się rozwija i jego znajomość jest niezbędna dla każdego danologa.

`Pandas` będzie dobrym wyborem do następujących zastosowań:
* Dane tabularyczne (kolumny jak w SQLu lub Excelu)
* Dane reprezentujące szeregi czasowe
* Macierze,
* Wyniki pomiarów i statystyk.

Dwa główne typy danych w Pythonie to `Series` (jednowymiarowa kolumna) i `DataFrame` (dwuwymiarowa tabela). `Pandas` wykorzystuje w obliczeniach bibliotekę `NumPy` oraz jest przygotowany do integrowania się z wieloma bibliotekami zewnętrznymi.

Mocnymi stronami `Pandas` są między innymi:
* Prosta obsługa brakujących wartości (`NaN`),
* Możliwość modyfikowania rozmiaru `DataFrame`'a - możemy dodawać i usuwać kolumny i wiersze,
* Automatyczne wyrównywanie danych w obliczeniach (jak w `NumPy`),
* Metoda `groupBy` działająca analogicznie jak w SQLu,
* Łatwo stworzyć `DataFrame` na podstawie innego obiektu,
* Cięcie, indeksowanie i tworzenie podzbiorów,
* Łączenie (`join` i `merge`) zbiorów.

***
# pandas w pigułce
"""

import numpy as np
import pandas as pd

pd.__version__

s = pd.Series([1,3,4,np.nan, 6, 8])
print(s)
dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
df2.dtypes

"""## Przeglądanie danych
Do przeglądania `DataFrame` służą między innymi następujące pola i metody.
"""

from IPython.display import display

display(df.head())
display(df.tail(3))

print(df.index)
print(df.columns)

"""## DataFrame.to_numpy()
`DataFrame.to_numpy()` jest metodą, która zamienia `DataFrame` na tablicę. Problemem jest to, że o ile `DataFrame` może przechowywać dane różnego typu, o tyle `ndarray` ma jeden `dtype` na całą tablicę. W związku z tym, może się okazać, że zajdzie konieczność castowania wszystkich obiektów na `object`.

`df.to_numpy()` będzie operacją błyskawiczną, natomiast `df2.to_numpy()` będzie już relatywnie wolne.
"""

# Commented out IPython magic to ensure Python compatibility.
# %timeit df.to_numpy()
# %timeit df2.to_numpy()

"""Dostępnych jest również trochę funkcji użytkowych (util)."""

print(df.info())  # informacje o DF
print('----')
display(df.describe())  # opis statystyczny
print('----')
display(df.T)  # zamiana wierszy z kolumnami
print('----')
display(df.sort_index(axis=1, ascending=False))  # sortowanie wg indeksu
print('----')
display(df.sort_values(by='B'))  # sortowanie według kolumny

"""## Pobieranie danych (select)"""

print(df['A']) # tylko kolumna A
print('----')
display(df[0:3]) # wiersze od 0 do 2
print('----')
display(df['20130102':'20130104'])  # od 2 do 3 stycznia
print('----')
print(df.loc[dates[0]])  # według wartości w indeksie
print('----')
display(df.loc[:, ['A', 'B']])  # wszystkie wiersze, ale tylko kolumny A i B
print('----')
display(df.loc['20130102':'20130104', ['A', 'B']])  # zakres wierszy
print('----')
print(df.loc['20130102', ['A', 'B']])  # tylko jeden wiersz
print('----')
print(df.loc[dates[0], 'A'])  # jedna komórka
print('----')
print(df.at[dates[0], 'A'])  # jedna komórka
print('----')
display(df.iloc[[3]])  # jeden wiersz, wg numeru wiersza jako DF
display(df.iloc[3])  # jako Series
print('----')
display(df.iloc[3:5, 0:2])  # według indeksów
print('----')
display(df[df.A < 0])  # indeksowanie warunkiem logicznym
print('----')
display(df[df > 0])  # szuka wartości mniejszych od zero

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
display(df2[df2['E'].isin(['two', 'four'])])  # wybór według wartości w komórkach

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1  # dodanie Series jako nowy wiersz
display(df)
print('----')

df.loc[:, 'D'] = np.array([5] * len(df))  # przypisanie numpy array
display(df)
print('----')

df[df > 0] = -df  # przypisanie ujemnych wartości tam gdzie są dodatnie
display(df)

"""## Praca z brakującymi danymi"""

display(df)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])  # wybierz nowy zakres indeksów z DF
display(df1)
df1.loc[dates[0]:dates[1], 'E'] = 1  # ustaw wartość w kolumnie E, w odpowiednich wierszach
display(df1)
df2 = df1.copy()

display(pd.isna(df1))  # sprawdź czy wartości są nan - do indeksowania warunkiem logicznym

display(df1.dropna(how='any'))  # usuń gdy nan jest gdziekolwiek; 'all' gdy tylko we wszystkich kolumnach
display(df2.fillna(value=5))  # wypełnij wartością value

"""## Operacje"""

print(df.mean())  # domyślnie - dla kolumn
print('----')
print(df.mean(axis=1))  # dla wierszy
print('----')
display(df.apply(np.cumsum))
print('----')
print(df.apply(lambda x: x.max() - x.min()))

"""## Łączenie i grupowanie DataFrame"""

df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]  # podział wg wierszy
print(pieces)
display(pd.concat(pieces))  # konkatenacja
print('----')

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
display(left)
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]}) 
display(right)
merged = pd.merge(left, right, on='key')  # łączenie wg wspólnej kolumny
display(merged)
print('----')

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
print(s)
df.append(s, ignore_index=True)  # dodaj wiersz, ignorując indeks
display(df)
print('----')


df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

display(df.groupby('A').sum())

display(df.groupby(['A', 'B']).sum())

"""## Cechy kategoryczne"""

df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
display(df)
print('----')
df["grade"] = df["raw_grade"].astype("category")  # konwersja do typu kategorii
display(df)
print('----')

df["grade"].cat.categories = ["very good", "good", "very bad"]  # ustawienie kategorii
display(df)
print('----')

display(df.sort_values(by="grade")) # sortowanie według zmiennej kategorycznej
print('----')
display(df.groupby("grade").size()) # grupowanie według wartości zmiennej kategorycznej

"""## Wizualizacje"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()  # zwykły wykres

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
df.plot()  #dla wielu wymiarów

"""## Wczytywanie `DataFrame` z pliku i zapis do pliku"""

df.to_csv('foo.csv')  # zapis do pliku

pd.read_csv('foo.csv')  # wczytywanie pliku
df.to_excel('foo.xlsx', sheet_name='Sheet1')  # zapis do formatu Excela

"""# Tutorial"""

!git clone https://github.com/matzim95/ML-datasets

"""**1. Najpierw importujemy niezbędne biblioteki**"""

import numpy as np
import pandas as pd  # konwencja

"""W pandasie mamy do czynienia z dwoma typami struktur: Series i Dataframes

**Series** to jednowymiarowa struktura danych (jednowymiarowa macierz numpy), która oprócz danych przechowuje też unikalny indeks. Taką serię możemy utworzyć następująco:
"""

pd.Series(np.random.random(10))

"""Drugą strukturą w pandas jest **DataFrame** - czyli dwu lub więcej wymiarowa struktura danych, najczęściej w formie tabeli z wierszami i kolumnami. Kolumny mają nazwy, a wiersze mają indeksy.

W tym szkoleniu skupimy się właśnie na DataFramach.

**2. Pierwszy krok jest zwykle ten sam. Dane są przechowywane w plikach csv, tsv, bazach danych, plikach excel itd. Wczytać je można np. z użyciem funkcji `pd.read_csv`**

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
"""

from IPython.display import display

# plik tsv - rozdzielony tabulatorami
chipotle = pd.read_csv('ML-datasets/chipotle.tsv', sep='\t')
# interesujące parametry: delimiter lub sep (alias), header (czy zawiera linię z opisem kolumn)
# names (lista kolumn), index_col (wybiera daną kolumnę jako indeks)
# usecols (wykorzystuje tylko określone kolumny)
display(chipotle)  # wyświetlamy - macierz z danymi różnych typów
display(chipotle.head())
display(chipotle.tail(3))
display(chipotle.sample(5))

"""Wczytamy teraz dataset `read_from`"""

read = pd.read_csv('ML-datasets/read_from.csv', sep=';',
                  names=['time', 'status', 'country', 'identifier', 'how', 'continent'],
                  parse_dates=True, index_col='time')

display(read.sample(5))

display(read[['country', 'how']])  # wybór wielu kolumn
display(read[['continent']])  # dataframe
display(read['continent'])  # Series
display(read.continent)  # Series

"""W pandasie istnieją co najmniej 4 metody pobierania danych. Pierwsza z nich to zwykłe nawiasy kwadratowe (jak wyżej). Można również w nawiasach podawać warunek logiczny."""

print(read.how == 'SEO')  # Series z wartościami True/False według wierszy
display(read[read.how == 'SEO'])  # Wyświetla wszystkie wiersze dla których było True

"""Teoretycznie, wiele operacji można wykonywać sekwencyjnie, gdyż większość z nich zwraca wynikową DataFrame."""

read.head()[['country', 'identifier']]

"""Możemy też wyświetlić indeks i listę kolumn"""

print(read.columns)
print(read.index)

"""Różne metody wyboru elementów """

dates = read.index
print(read['identifier']) # tylko kolumna A
print('----')
display(read[0:3]) # wiersze od 0 do 2
print('----')
display(read['2018-01-01 12:00':'2018-01-01 12:10'])
print('----')
display(read.loc[:, ['country', 'continent']])  # wszystkie wiersze, ale tylko kolumny A i B
print('----')
display(read.loc[dates[0], 'country'])
print('----')
display(read.loc['2018-01-01 12:00':'2018-01-01 12:10', ['country', 'continent']])  # zakres wierszy
print('----')
print(read.loc['2018-01-01 12:00', ['country', 'continent']])  # tylko jeden wiersz
print('----')
print(read.loc[dates[0], 'country'])  # jedna komórka
print('----')
print(read.at[dates[0], 'country'])  # jedna komórka
print('----')
display(read.iloc[[3]])  # jeden wiersz, wg numeru wiersza jako DF
display(read.iloc[3])  # jako Series
print('----')
display(read.iloc[3:5, 0:2])  # według indeksów
print('----')
display(read[read.country == 'country_2'])  # indeksowanie warunkiem logicznym
print('----')
display(read[read != 'Asia'])  # szuka wartości różnych od ''

"""Porównanie czasu działania numpy i pandas"""

# Import numpy
import numpy as np

iris = pd.read_csv('ML-datasets/iris.csv')
iris.pop('species')
f1 = lambda x: np.log10(x.to_numpy())
f2 = lambda x: np.log10(x)

# Commented out IPython magic to ensure Python compatibility.
# Create array of DataFrame values: np_vals with log10
# %timeit f1(iris)

# Create array of new DataFrame by passing df to np.log10(): df_log10
# %timeit f2(iris)

"""## Agregacja danych
Jest to proces łączenia wartości ze zbioru danych (lub jego podzbioru) w jedną wartość. Przykładowo, mając listę samochodów można zagregować kolumnę ceny do łącznej wartości wszystkich samochodów.
"""

import numpy as np
import pandas as pd

cars = pd.read_csv('ML-datasets/auto_mpg.csv', na_values='?')
display(cars)

"""Metoda `count` służy do zliczania liczby elementów w Series. Dlatego, też dla całego DataFrame zwróci wartości dla poszczególnych kolumn."""

print(cars.count())

print('----')

# aby uzyskać jedną wartość - wystarczy wybrać jedną kolumnę
print(cars[['mpg']].count())
print(cars.count().mpg)
print(cars.mpg.count())

"""Możemy łatwo zsumować wszystkie wartości, zarówno w kolumnie, jak i macierzy, ustalić wartość maksymalną i minimalną, średnią czy też medianę."""

print(cars.weight.sum())
print(cars.sum())
print('----')
print(cars.cylinders.min())
print(cars.min())
print('----')
print(cars.acceleration.max())
print(cars.max())
print('----')
print(cars.modelyear.mean())
print(cars.mean())
print('----')
print(cars.mpg.median())
print(cars.median())

"""Jako Data Scientists od czasu do czasu trzeba wykonać segmentację bazy danych. Oprócz wyznaczania statystyk dla wszystkich wartości, czasem można te wartości pogrupować. W pandasie służy do tego metoda groupby."""

display(cars.groupby('cylinders').mean().horsepower)
display(cars.groupby('modelyear').max()[['acceleration']])

"""W jaki sposób najczęście czytano w zbiorze `read`?"""

read.groupby('how').count()[['identifier']]

"""Jaka była najczęstsza kombinacja źródła i tematu dla kraju `country_2`?"""

read[read.country == 'country_2'].groupby(['how', 'continent']).count()[['identifier']]

"""## Łączenie dataframes

W rzeczywistości często nie chcemy korzystać z jednej dużej bazy, lecz łączymy wiele mniejszych (łatwiej jest nimi zarządzać, unikać redundancji, dodatkowo oczędzamy miejsce na dysku i osiągamy większą szybkość.

W Pandasie do łączenia dwóch tabel wykorzystujemy funkcję `merge`, która w swoich założeniach jest bardzo podobna do SQL-owego JOINa.
"""

import pandas as pd
import numpy as np

zoo = pd.read_csv('ML-datasets/zoo.csv')
zoo.animal.unique()

zoo_eats = pd.DataFrame({'animal': ['elephant', 'tiger', 'zebra', 'giraffe', 'kangaroo'],
                         'food': ['vegetables', 'meat', 'vegetables', 'vegetables', 'vegetables']})
zoo_eats

zoo.merge(zoo_eats)  # zniknęły lwy, bo nie było ich w zoo_eats

zoo_eats.merge(zoo)  # inna kolejność kolumn

# strategie merge:
# inner
zoo.merge(zoo_eats, how='inner')  # domyślne: część wspólna kolumn

# outer
zoo.merge(zoo_eats, how='outer')  # pojawiły się nany - suma

# left będą dla wartości z zoo
zoo.merge(zoo_eats, how='left')

# right - odwrotnie
zoo.merge(zoo_eats, how='right')

# wykorzystywana kolumna
# domyślnie pandas próbuje sam znaleźć, ale często trzeba to zrobić ręcznie
zoo.merge(zoo_eats, how='outer', left_on='animal', right_on='animal')

"""Sortowanie w Pandasie"""

cars.sort_values('cylinders')

# sortowanie po wielu kolumnach, według kolejności podania
cars.sort_values(by=['cylinders', 'mpg'])

# sortowanie malejące
cars.sort_values('horsepower', ascending=False)

# można by było zresetować indeksy
cars.sort_values('horsepower', ascending=False).reset_index()
# nie tylko że są brzydkie, ale mogą też pomieszać dostęp według numeru indeksu, czy też wizualizacje

# usuwanie starego indeksu
cars.sort_values('horsepower', ascending=False).reset_index(drop=True)

# radzenie sobie z wartościami NaN
# wypełnianie wartości NaN
zoo.merge(zoo_eats, how='left').fillna('meat')

# dropna
zoo.merge(zoo_eats, how='right').dropna()

"""# Zadania

**1. Porównywanie pd.Series**

Utwórz dwa obiekty typu `pd.Series`, zawierające 100 wartości 0 lub 1. Wykorzystaj w tym celu generator liczb losowych z biblioteki NumPy [np.random.randint](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html#numpy-random-randint).

Następnie wyświetl wszystkie indeksy, na których wartości w obu Series się zgadzają.
"""

import numpy as np
import pandas as pd

a = pd.Series(np.random.randint(2, size=100))
b = pd.Series(np.random.randint(2, size=100))
a[a == b].index

"""**2. Zbiór chipotle - kto zapłacił najwięcej?**

Wczytaj zbiór chipotle i dodaj kolumne `price_dollars`, zawierającą cenę w dolarach. Następnie sprawdź, które z zamówień opiewało na najwyższą kwotę (wykorzystaj w tym celu grupowanie i wybraną agregację). Podaj jego indeks (`order_id`) oraz wartość zamówienia.
"""

import pandas as pd

dolarizer = lambda x: float(x[1:])

chipotle = pd.read_csv('ML-datasets/chipotle.tsv', sep='\t')
chipotle['price_dollars'] = chipotle.item_price.apply(dolarizer)
price_summed = chipotle.groupby('order_id').price_dollars.sum()
print(f'Order {price_summed.idxmax()}: {price_summed.max()}')

"""**3. Łączenie DataFrames**

Wczytaj zbiory danych `rating.csv` oraz `parking.csv`. Spróbuj połączyć je wykorzystując metodę `merge`. Pamiętaj o odpowiedniej nazwie kol
"""

import pandas as pd

rating = pd.read_csv('ML-datasets/rating.csv')
parking = pd.read_csv('ML-datasets/parking.csv')

merged = rating.merge(parking, how='left', on='placeID')
merged.sample(5)

a
