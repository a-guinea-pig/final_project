import streamlit as st
import pandas as pd
import numpy as np

with st.echo(code_location='below'):

    st.write("эта страница состоит из двух блоков: данные по ценам на жильё в Москве и данные по МЦД")

    st.subheader('wены на жильё')

    st.write("в этом блоке используется pandas для подготовки данных, предсказательные модели sklearn, "
             "визуализации plotly express и немного numpy для статистики"
             "используемый датасет: https://www.kaggle.com/datasets/hishamhaydar/moscow-2018-housing-prices")

    # импортируем данные и уберём некоторые колонки: пустые, нумерацию, а также год постройки — там много пустых значений.
    # заполним нулями ячейки в столбцах-индикаторах первого и второго этажа
    # потом почистим собственно строчки с пустыми значениями

    housing_data_raw = pd.read_excel("2_5393538523506672609.xlsx")
    housing_data_raw = housing_data_raw.fillna({'floor1': 0, 'floor2': 0})
    housing_data = housing_data_raw.drop(columns=['num', 'N', 'Tel', 'built']).dropna()

    st.subheader("немного статистики:")
    price_array = housing_data[["Price"]].to_numpy()
    sq_array = housing_data[["Totsp"]].to_numpy()
    st.write("медианная цена — " + str(np.median(price_array)))
    st.write("медианная площадь — " + str(np.median(sq_array)))

    # теперь будем строить модельки, которые учатся на этих данных предсказывать стоимость квартиры по её параметрам.
    # разделим исходный датасет на два и посмотрим, какая модель лучше предскажет стоимость

    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    train_housing_data, test_housing_data = train_test_split(housing_data)
    linear_model = LinearRegression()
    linear_model.fit(train_housing_data.drop(columns=["Price"]), train_housing_data["Price"])
    knn_model = KNeighborsRegressor()
    knn_model.fit(train_housing_data.drop(columns=["Price"]), train_housing_data["Price"])

    #linear_model.coef_

    st.subheader("регрессии")

    st.write("построим две регрессии — линейную и по ближайшим соседям")

    test_prediction_linear = pd.DataFrame({'Totsp': np.array(test_housing_data['Totsp']), 'Price':linear_model.predict(test_housing_data.drop(columns =['Price']))})

    test_prediction_knn = pd.DataFrame({'Totsp': np.array(test_housing_data['Totsp']), 'Price':knn_model.predict(test_housing_data.drop(columns =['Price']))})

    st.write("посмотрим на картиночки на тестовом наборе данных")

    import plotly.express as px

    fig0 = px.scatter(train_housing_data, x = 'Totsp', y = 'Price', color_discrete_sequence=px.colors.qualitative.Dark2, opacity=0.5)
    fig0.add_trace(px.scatter(test_housing_data, x = 'Totsp', y = 'Price', opacity=0.5).data[0])
    st.plotly_chart(fig0)

    import matplotlib.pyplot as plt


    select_regression = st.selectbox( label = "выбираем регрессию", options = ["linear", "knn"])


    if select_regression == "linear":
        fig1 = px.scatter(test_housing_data, x='Totsp', y='Price', color_discrete_sequence=px.colors.qualitative.Dark2,
                          opacity=0.5)
        fig1.add_trace(px.scatter(test_prediction_linear, x='Totsp', y='Price', opacity=0.5).data[0])
        st.plotly_chart(fig1);
        sd_linear = ((test_housing_data["Price"] / 1000000 - test_prediction_linear["Price"] / 1000000) ** 2).mean()
        st.write('квадратное отклонение стоимости (в миллионах рублей) — ' + str(sd_linear))

    elif select_regression == "knn":
        fig2 = px.scatter(test_housing_data, x='Totsp', y='Price', color_discrete_sequence=px.colors.qualitative.Dark2,
                          opacity=0.5)
        fig2.add_trace(px.scatter(test_prediction_knn, x='Totsp', y='Price', opacity=0.5).data[0]);
        st.plotly_chart(fig2)
        sd_knn = ((test_housing_data["Price"] / 1000000 - test_prediction_knn["Price"] / 1000000) ** 2).mean()
        st.write('квадратное отклонение стоимости (в миллионах рублей) — ' + str(sd_knn))
    # посчитаем отклонения в миллионах рублей

    st.write("отклонение примерно одинаковое, но линейная регрессия немного точнее.")

    rooms = st.number_input('количество комнат')
    totsp = st.number_input('общая площадь')
    livesp = st.number_input('жилая площадь')
    kitsp = st.number_input('площадь кухни')
    dist = st.number_input('расстояние от центра города')
    metrdist = st.number_input('расстояние до метро')
    walk = st.number_input(label = '1, если до метро можно дойти пешком, 0 — если нужно ехать на транспорте', min_value = 0, max_value = 1, step = 1)
    brick = st.number_input(label = '1, если дом кирпичный, 0 — иначе', min_value = 0, max_value = 1, step = 1)
    bal = st.number_input(label = '1, если есть балкон, 0 — иначе', min_value = 0, max_value = 1, step = 1)
    floor = st.number_input(label = '1, если квартира находится не на первом или последнем этаже, 0 — иначе', min_value = 0, max_value = 1, step = 1)
    new = st.number_input(label = '1, если новостройка, 0 — иначе', min_value = 0, max_value = 1, step = 1)
    floors = st.number_input('количество этажей в доме')
    nfloor = st.number_input('этаж, на котором расположена квартира')
    floor1 = st.number_input(label = '1, если квартира на первом этаже, 0 — иначе', min_value = 0, max_value = 1, step = 1)
    floor2 = st.number_input(label = '1, если квартира на последнем этаже, 0 — иначе', min_value = 0, max_value = 1, step = 1)
    rooms2 = rooms

    columns_list = list(housing_data)
    columns_list.remove('Price')
    len(columns_list)

    linear_pred_price = linear_model.predict(pd.DataFrame([[rooms,
                                      totsp,
                                      livesp,
                                      kitsp,
                                      dist,
                                      metrdist,
                                      walk,
                                      brick,
                                      bal,
                                      floor,
                                      new,
                                      floors,
                                      nfloor,
                                      floor1,
                                      floor2,
                                      rooms2]], columns = columns_list))

    st.write("predicted price: " + str(linear_pred_price[0]))

    st.subheader('МЦД')

    st.write("в этом блоке используется sql и pandas для работы с данными, geopandas и folium для визуализации "
             "размещения станций; данные взяты в формате json"
             "данные с https://apidata.mos.ru/help/index#!/Features/Features_GetListByDatasetId_0, id 62207")

    import json
    with open('map.geojson', encoding = 'utf-8') as mcd_data_json:
        mcd_data = json.load(mcd_data_json)
    mcd_StationNames_list = []
    mcd_DiameterNames_list = []
    mcd_Latitude_WGS84_list = []
    mcd_Longitude_WGS84_list = []
    mcd_global_id_list = []

    for i in range(0, len(mcd_data['features'])):
        mcd_StationNames_list.append(mcd_data['features'][i]['properties']['Attributes']['Diameter'][0]['StationName'])
        mcd_DiameterNames_list.append(mcd_data['features'][i]['properties']['Attributes']['Diameter'][0]['DiameterName'])
        mcd_Latitude_WGS84_list.append(mcd_data['features'][i]['properties']['Attributes']['Latitude_WGS84'])
        mcd_Longitude_WGS84_list.append(mcd_data['features'][i]['properties']['Attributes']['Longitude_WGS84'])
        mcd_global_id_list.append(mcd_data['features'][i]['properties']['Attributes']['global_id'])

    mcd_df_data = zip(mcd_global_id_list, mcd_StationNames_list, mcd_DiameterNames_list, mcd_Latitude_WGS84_list,
                      mcd_Longitude_WGS84_list)
    mcd_df = pd.DataFrame(mcd_df_data, columns=["exit id", "station name", "diameter", "lat", "lon"])

    station_name = st.text_input('Напишите имя любой станции МЦД, чтобы посмотреть, к какому диаметру'
                                 'она относится и сколько выходов есть на этой станции', "Щукинская")
    number_of_exits = mcd_df.groupby('station name').count().loc[station_name]['exit id']
    diam_name = mcd_df.loc[mcd_df['station name'] == station_name]['diameter'].iloc[0]
    st.write('Эта станция относится к ' + diam_name + ', число выходов — ' + str(number_of_exits))

    from sqlalchemy import create_engine
    engine = create_engine('sqlite://', echo=False)

    mcd_df.to_sql('mcd', con=engine)

    mcd_grouped_by_station = engine.execute("""
    SELECT `station name`, avg(lat) as avg_lat, avg(lon) as avg_lon, diameter as diameter FROM mcd
    GROUP BY `station name`
    """).fetchall()

    engine.execute("""
    PRAGMA table_info(mcd);
    """).fetchall()

    mcd1 = engine.execute("""
    SELECT `station name`, avg(lat) as avg_lat, avg(lon) as avg_lon, diameter as diameter FROM mcd
    GROUP BY `station name`
    HAVING diameter == 'МЦД-1'
    """).fetchall()

    mcd2 = engine.execute("""
    SELECT `station name`, avg(lat) as avg_lat, avg(lon) as avg_lon, diameter as diameter FROM mcd
    GROUP BY `station name`
    HAVING diameter == 'МЦД-2'
    """).fetchall()

    # вернём сделанные в sql таблички обратно в pandas -> geopandas для визуализации

    mcd1_df = pd.DataFrame(mcd1, columns = ['station name', 'lat', 'lon', 'diam'])
    mcd2_df = pd.DataFrame(mcd2, columns = ['station name', 'lat', 'lon', 'diam'])



    from shapely.geometry import Polygon

    # загрузим полигоны районов Москвы

    with open('mo.geojson', encoding = 'utf-8') as mo_data:
        mo_json = json.load(mo_data)

    # загрузим дополнительно городские округа Московской области, в которых есть МЦД

    import requests

    regions = ['Долгопрудный', 'Лобня', 'Красногорск', 'Одинцовский городской округ', 'Ленинский городской округ']
    regions_id = []
    for region in regions:
        entrypoint = "https://nominatim.openstreetmap.org/search"
        params = {'q': region,

                  'limit': 1,
                  'format': 'json'}
        resp_mr = requests.get(entrypoint, params=params)
        data_mr = resp_mr.json()
        for item in data_mr:
            relation_id = item["osm_id"]
        regions_id.append(relation_id)

    regions_polygons = []
    for i in range(0, 5):
        entrypoint_p = "https://nominatim.openstreetmap.org/reverse?format=json&osm_id=" + str(
            regions_id[i]) + "&osm_type=R&polygon_geojson=1"
        params_p = {'format': 'json'}
        resp_p = requests.get(entrypoint_p, params=params_p)
        regions_polygons.append(resp_p.json())

    names0_list = regions
    polys0_list = []
    for i in range(0, 2):
        poly0 = Polygon(regions_polygons[i]['geojson']['coordinates'][0])
        polys0_list.append(poly0)

    poly0 = Polygon(regions_polygons[2]['geojson']['coordinates'][0][0])
    polys0_list.append(poly0)

    for i in range(3, 5):
        poly0 = Polygon(regions_polygons[i]['geojson']['coordinates'][0])
        polys0_list.append(poly0)

    names1_list = []
    polys1_list = []
    for reg in mo_json['features']:
        try:
            name1 = reg['properties']['NAME']
            poly1 = Polygon(reg['geometry']['coordinates'][0])
            names1_list.append(name1)
            polys1_list.append(poly1)
        except ValueError:
            name1 = reg['properties']['NAME']
            poly1 = Polygon(reg['geometry']['coordinates'][0][0])
            names1_list.append(name1)
            polys1_list.append(poly1)

    # используем geopandas для того, чтобы распределить станции по районам Москвы и области
    # geopandas не работает в streamlit, поэтому выполним это в jupiter notebook и загрузим сюда готовые данные

    # import geopandas as gpd

    # data1 = zip(names1_list + names0_list, polys1_list + polys0_list)
    # gdf1 = gpd.GeoDataFrame(pd.DataFrame(data1, columns=["name", "poly"]), geometry='poly')
    # json_df1 = gdf1.to_json()

    # mcd1_gdf = gpd.GeoDataFrame(mcd1_df, geometry = gpd.points_from_xy(mcd1_df['lon'], mcd1_df['lat']))
    # mcd2_gdf = gpd.GeoDataFrame(mcd2_df, geometry = gpd.points_from_xy(mcd2_df['lon'], mcd2_df['lat']))

    # stations_counts_2 = gpd.sjoin(gdf1, mcd2_gdf, how='inner', predicate="intersects")["name"].value_counts()
    # stations_counts_1 = gpd.sjoin(gdf1, mcd1_gdf, how='inner', predicate="intersects")["name"].value_counts()

    # n_by_distr1 = pd.DataFrame(zip(list(stations_counts_1.to_frame().T), stations_counts_1.to_list()), columns = ['name', 'N'])
    # n_by_distr2 = pd.DataFrame(zip(list(stations_counts_2.to_frame().T), stations_counts_2.to_list()), columns = ['name', 'N'])

    # n_by_distr1.to_csv(r'/Users/---/Downloads/n_by_distr1.csv', sep=',', index = False)
    # n_by_distr2.to_csv(r'/Users/---/Downloads/n_by_distr2.csv', sep=',', index = False)

    with open('json_df1.json', encoding = 'utf-8') as json_df1_data:
        json_df1 = json.load(json_df1_data)

    n_by_distr1 = pd.read_csv('n_by_distr1.csv')
    n_by_distr2 = pd.read_csv('n_by_distr2.csv')

    from streamlit_folium import folium_static
    import folium

    st.subheader('Распределение станций МЦД-1')

    m1 = folium.Map(location=[55.7, 37.6], zoom_start=9)

    folium.Choropleth(geo_data = json_df1,
                     name = 'choropleth',
                     data = n_by_distr1,
                     columns = ['name', 'N'],
                     key_on="feature.properties.name",
                     fill_color="YlGnBu",
                     fill_opacity=1,
                     line_opacity=0.2).add_to(m1)

    folium.LayerControl().add_to(m1)

    folium_static(m1)

    st.subheader('Распределение станций МЦД-2')

    m2 = folium.Map(location=[55.7, 37.6], zoom_start=9)

    folium.Choropleth(geo_data = json_df1,
                     name = 'choropleth',
                     data = n_by_distr2,
                     columns = ['name', 'N'],
                     key_on="feature.properties.name",
                     fill_color="YlGnBu",
                     fill_opacity=1,
                     line_opacity=0.2).add_to(m2)

    folium.LayerControl().add_to(m2)

    folium_static(m2)

