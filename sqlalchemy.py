


"""Wykonaj tylko raz, nie resetuje się wraz ze zmiennymi środowiskowymi"""

!apt-get install mysql-server > /dev/null
!service mysql start
!mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'root'"
!pip -q install PyMySQL

from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:root@/")
engine.execute("CREATE DATABASE car_rental") #create db
engine.execute("USE car_rental") # select new db

"""Pierwszy blok zadaniowy"""

Stwórz tabele *cars, clients, bookings* według wytycznych(bez relacji):

*   cars: car_id(int, pk), producer(str), model(str), year(int), horse_power(int), price_per_day(int)
*   clients: client_id(int, pk), name(str), surname(str), address(str), city(str)
*   bookings: booking_id(int, pk), client_id(int), car_id(int), start_date(date), end_date(date), total_amount(int)


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Date

eng = create_engine("mysql+pymysql://root:root@/car_rental", echo = True)

base = declarative_base()

class Cars(base):
   __tablename__ = 'cars'

   car_id = Column(Integer, primary_key=True, autoincrement=True)
   producer = Column(String(30), nullable=False)
   model = Column(String(30), nullable=False)
   year = Column(Integer, nullable=False)
   horse_power = Column(Integer, nullable=False)
   price_per_day = Column(Integer, nullable=False)

class Clients(base):
    __tablename__ = 'clients'

    client_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=False)
    surname = Column(String(30), nullable=False)
    address = Column(String(30), nullable=False)
    city = Column(String(30), nullable=False)

class Bookings(base):
    __tablename__ = 'bookings'

    booking_id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, nullable=False)
    car_id = Column(Integer, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    total_amount = Column(Integer, nullable=False)

base.metadata.create_all(eng)

"""To samo zadanie można wykonać za pomocą odpowiednich zapytań SQLa:"""

sql_statement = 'CREATE TABLE sql_booking(booking_id INTEGER PRIMARY KEY,' \
'client_id INTEGER,car_id INTEGER,start_date DATE,end_date DATE,total_amount INTEGER);'

eng.execute(sql_statement)

# Check
result = eng.execute("SHOW TABLES;")
for description in result:
    print(description)

sql_statement = 'DROP TABLE sql_booking'
eng.execute(sql_statement)

"""Kolejne zadanie to dodanie kilku instancji do naszej tabeli"""

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=eng)
session = Session()
client_1 = Clients(name='Jan', surname='Kowalski', address='ul. Florianska 12', city='Krakow')
car_1 = Cars(producer='Seat', model='Leon', year=2016, horse_power=80, price_per_day=200)

session.add(client_1)
session.add(car_1)
session.commit()

"""W kolejnym należy wyświetlić zawartość poszczególnych tabel"""

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=eng)
session = Session()

for client in session.query(Clients).all():
    print(client)

for car in session.query(Cars).all():
    print(car)

"""W celu zwiększenia czytelności warto zdefiniować reprezentację stringową naszych obiektów."""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Date

eng = create_engine("mysql+pymysql://root:root@/car_rental", echo = True)

base = declarative_base()

class Cars(base):
   __tablename__ = 'cars'

   car_id = Column(Integer, primary_key=True, autoincrement=True)
   producer = Column(String(30), nullable=False)
   model = Column(String(30), nullable=False)
   year = Column(Integer, nullable=False)
   horse_power = Column(Integer, nullable=False)
   price_per_day = Column(Integer, nullable=False)

   def __repr__(self):
       return f'<Car: id={self.car_id}, producer={self.producer},' \
              f'model={self.model}, year={self.year},' \
              f'horse_power={self.horse_power}, price_per_day={self.price_per_day}>'


class Clients(base):
    __tablename__ = 'clients'

    client_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=False)
    surname = Column(String(30), nullable=False)
    address = Column(String(30), nullable=False)
    city = Column(String(30), nullable=False)

class Bookings(base):
    __tablename__ = 'bookings'

    booking_id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, nullable=False)
    car_id = Column(Integer, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    total_amount = Column(Integer, nullable=False)

base.metadata.create_all(eng)

"""Teraz uzyskamy troszkę bardziej czytelne wyniki:

Przy okazji w taki sposób można wykorzystać select
"""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

Session = sessionmaker(bind=eng)
session = Session()

s = select([Clients]) # Make sql_statement
print(s)

for row in eng.execute(s):
	print(row)

for car in session.query(Cars).all():
    print(car)

"""Stwórzmy teraz możliwość dodawania dużej ilośc danych poprzez funkcje insert_data"""

def insert_data(session, base, params):
    session.add(base(**params))
    session.commit()

clients = [
    {'name': 'Jan', 'surname': 'Kowalski', 'address': 'ul. Florianska 12', 'city': 'Krakow'},
    {'name': 'Andrzej', 'surname': 'Nowak', 'address': 'ul. Saska 43', 'city': 'Wroclaw'},
    {'name': 'Michal', 'surname': 'Taki', 'address': 'os. Srodkowe 12', 'city': 'Poznan'},
    {'name': 'Pawel', 'surname': 'Ktory', 'address': 'ul. Stara 11', 'city': 'Gdynia'},
    {'name': 'Anna', 'surname': 'Inna', 'address': 'os. Srednie 1', 'city': 'Gniezno'},
    {'name': 'Alicja', 'surname': 'Panna', 'address': 'os. Duze 33', 'city': 'Torun'},
    {'name': 'Damian', 'surname': 'Papa', 'address': 'ul. Skosna 66', 'city': 'Warszawa'},
    {'name': 'Marek', 'surname': 'Troska', 'address': 'os. Male 90', 'city': 'Radom'},
    {'name': 'Jakub', 'surname': 'Klos', 'address': 'os. Polskie 19', 'city': 'Wadowice'},
    {'name': 'Lukasz', 'surname': 'Lis', 'address': 'os. Podlaskie 90', 'city': 'Bialystok'}]
cars = [
    {'producer': 'Seat', 'model': 'Leon', 'year': 2016, 'horse_power': 80, 'price_per_day': 200},
    {'producer': 'Toyota', 'model': 'Avensis', 'year': 2014, 'horse_power': 72, 'price_per_day': 100},
    {'producer': 'Mercedes', 'model': 'CLK', 'year': 2018, 'horse_power': 190, 'price_per_day': 400},
    {'producer': 'Hyundai', 'model': 'Coupe', 'year': 2014, 'horse_power': 165, 'price_per_day': 300},
    {'producer': 'Dacia', 'model': 'Logan', 'year': 2015, 'horse_power': 103, 'price_per_day': 150},
    {'producer': 'Saab', 'model': '95', 'year': 2012, 'horse_power': 140, 'price_per_day': 140},
    {'producer': 'BMW', 'model': 'E36', 'year': 2007, 'horse_power': 110, 'price_per_day': 80},
    {'producer': 'Fiat', 'model': 'Panda', 'year': 2016, 'horse_power': 77, 'price_per_day': 190},
    {'producer': 'Honda', 'model': 'Civic', 'year': 2019, 'horse_power': 130, 'price_per_day': 360},
    {'producer': 'Volvo', 'model': 'XC70', 'year': 2013, 'horse_power': 180, 'price_per_day': 280}]
bookings = [
    {'client_id': 3, 'car_id': 3, 'start_date': '2020-07-06', 'end_date': '2020-07-08', 'total_amount': 400},
    {'client_id': 6, 'car_id': 10, 'start_date': '2020-07-10', 'end_date': '2020-07-16', 'total_amount': 1680},
    {'client_id': 4, 'car_id': 5, 'start_date': '2020-07-11', 'end_date': '2020-07-14', 'total_amount': 450},
    {'client_id': 5, 'car_id': 4, 'start_date': '2020-07-11', 'end_date': '2020-07-13', 'total_amount': 600},
    {'client_id': 7, 'car_id': 3, 'start_date': '2020-07-12', 'end_date': '2020-07-14', 'total_amount': 800},
    {'client_id': 5, 'car_id': 7, 'start_date': '2020-07-14', 'end_date': '2020-07-17', 'total_amount': 240},
    {'client_id': 3, 'car_id': 8, 'start_date': '2020-07-14', 'end_date': '2020-07-16', 'total_amount': 380},
    {'client_id': 5, 'car_id': 9, 'start_date': '2020-07-15', 'end_date': '2020-07-18', 'total_amount': 1080},
    {'client_id': 6, 'car_id': 10, 'start_date': '2020-07-16', 'end_date': '2020-07-20', 'total_amount': 1120},
    {'client_id': 8, 'car_id': 1, 'start_date': '2020-07-16', 'end_date': '2020-07-19', 'total_amount': 600},
    {'client_id': 9, 'car_id': 2, 'start_date': '2020-07-16', 'end_date': '2020-07-21', 'total_amount': 500},
    {'client_id': 10, 'car_id': 6, 'start_date': '2020-07-17', 'end_date': '2020-07-19', 'total_amount': 280},
    {'client_id': 1, 'car_id': 9, 'start_date': '2020-07-17', 'end_date': '2020-07-19', 'total_amount': 720},
    {'client_id': 3, 'car_id': 7, 'start_date': '2020-07-18', 'end_date': '2020-07-21', 'total_amount': 240},
    {'client_id': 5, 'car_id': 4, 'start_date': '2020-07-18', 'end_date': '2020-07-22', 'total_amount': 1200}]

for client in clients:
    insert_data(session, Clients, client)
for car in cars:
    insert_data(session, Cars, car)
for booking in bookings:
    insert_data(session, Bookings, booking)

"""Spróbujmy teraz podziałać na tej utworzonej bazie:"""

#1
result = session.query(Bookings).filter(Bookings.client_id == 3)
for booking in result:
    print(booking)

#2
from sqlalchemy.sql import select

conn = eng.connect()
s = select([Bookings]).where(Bookings.client_id == 3)
result = conn.execute(s).fetchall()
print(result)

from sqlalchemy.sql import select
from sqlalchemy import join

j = join(Bookings, Cars, Bookings.car_id == Cars.car_id)
s = select([Cars]).select_from(j).where(Bookings.client_id == 5)
result = conn.execute(s)
for car in result:
    print(car)

import pickle

#Here's an example dict
grades = { 'Alice': 89, 'Bob': 72, 'Charles': 87 }

#Use dumps to convert the object to a serialized string
serial_grades = pickle.dumps( grades )
print(serial_grades)
#Use loads to de-serialize an object
received_grades = pickle.loads( serial_grades )
print(received_grades)

import sys
!pip install -U pip
!{sys.executable} -m pip install -U pandas-profiling[notebook]
!jupyter nbextension enable --py widgetsnbextension

from google.colab import drive
drive.mount('/drive')

import os
os.chdir("drive")
os.chdir("My Drive")

!ls

# Standard Library Imports
from pathlib import Path

# Installed packages
import pandas as pd
from ipywidgets import widgets

# Our package
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file

# Read the Titanic Dataset
file_name = cache_file(
    "titanic.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
)
df = pd.read_csv(file_name)

# Generate the Profiling Report
profile = ProfileReport(df, title="Titanic Dataset", html={'style': {'full_width': True}}, sort="None")

# The Notebook Widgets Interface
profile.to_file('report.html')
