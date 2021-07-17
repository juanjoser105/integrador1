from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse

#Importaciones del proyecto
from bs4 import BeautifulSoup
import csv
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import gensim
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Librerías necesarias para abrir imágenes, generar nube de palabras y plot
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

#Librerías necesarias para la limpieza de datos
import string
import nltk
from nltk.corpus import stopwords
from tkinter import *
from PIL import ImageTk, Image

#Create your views here

def busqueda_ppl(request):

    return render(request, "busqueda_ppl.html")

def buscar(request):

    if request.GET["prd"]:
        parametro="%r" %request.GET["prd"]
        """
        #Ingreso del idioma de búsqueda
        idioma = "ES"

        #Ingreso la cantidad que quiero de resultados (Multiplo de 10)
        cantidadres = 10
        contador = 0

        #Ingreso del parámetro de búsqueda
        subcadena = parametro.split(',')
        cadena = subcadena[0]

        #Reemplazo de espacios blancos por el signo +
        nueva_cadena = cadena.replace(" ", "+")
        #Crear nuevo archivo
        file = open('resultados.csv','w',encoding="utf-8")
        writer = csv.writer(file)
    
    
        #Escribir las cabeceras del archivo
        writer.writerow(['titulo','vinculo','resumen'])
    
        #Cabecera del navegador
        #headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'}
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36 Edg/89.0.774.77'}
    
        #URL de acceso a los resultados de busqueda
        url = 'https://scholar.google.com/scholar?hl='+idioma+'&q='+nueva_cadena
    
        #Respuesta obtenida
        response = requests.get(url, headers=headers)
    
        #creacion objeto BeaustifulSoup con los parametros requeridos para el webscraping
        soup= BeautifulSoup(response.content,'lxml')
    
        itbusqueda = cantidadres/10
        for i in range(0, int(itbusqueda)):
            start = i*10
            
            #URL de acceso a los resultados de busqueda
            url2 = 'https://scholar.google.com/scholar?hl='+idioma+'&q='+nueva_cadena+'&start='+str(start)
            
            #Respuesta obtenida
            response = requests.get(url2, headers=headers)
    
            #creacion objeto BeaustifulSoup con los parametros requeridos para el webscraping
            soup= BeautifulSoup(response.content,'lxml')
    
            for item in soup.select('[data-lid]'):
                #print(item)
                #Titulo del articulo
                titulo = str(item.select('h3')[0].get_text())
    
                tipodoc = titulo.find("[CITAS]")
    
                if tipodoc==-1:
                    #vinculo del articulo
                    vinculo = str(item.select('a')[0]['href'])
                    print(vinculo)
                    #Resumen
                    resumen = str(item.select('.gs_rs')[0].get_text())
                    print(resumen)
                    print('-------------------')
                    writer.writerow([str(titulo),str(vinculo),str(resumen)]) 
                
            itbusqueda = int(itbusqueda) - 1
                
        file.close()
        
        #Analisis archivo csv
        data = pd.read_csv('resultados.csv') 
        
        tfidf = TfidfVectorizer(
            min_df = 5,
            max_df = 0.95,
            max_features = 8000,
            stop_words = 'english'
        )
        tfidf.fit(data.resumen)
        text = tfidf.transform(data.resumen)
        
        def find_optimal_clusters(data, max_k):
            iters = range(2, max_k+1, 2)
            
            sse = []
            for k in iters:
                sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
                print('Fit {} clusters'.format(k))
                
            f, ax = plt.subplots(1, 1)
            ax.plot(iters, sse, marker='o')
            ax.set_xlabel('Cluster Centers')
            ax.set_xticks(iters)
            ax.set_xticklabels(iters)
            ax.set_ylabel('SSE')
            ax.set_title('SSE by Cluster Center Plot')
            plt.savefig('example.jpg')
            
        find_optimal_clusters(text, 10)
        
        global my_img
        top = Toplevel()
        top.title('Analisis graph')
        my_img = ImageTk.PhotoImage(Image.open("example.jpg"))
        my_label = Label(top, image=my_img)
        my_label.pack()
            
        """




        return render(request, "analisis.html", {"busqueda": parametro})
    else:
        mensaje="no haz introducido ningun valor de busqueda"
    return HttpResponse(mensaje)

def analisis(request):
    if request.GET["codo"]:
        codo=int(request.GET["codo"])
        contexto = str(request.GET["contexto"])
        
        data = pd.read_csv('resultados.csv') 
        contador = len(data)
        print(contador)
        tfidf = TfidfVectorizer(
                min_df = 5,
                max_df = 0.95,
                max_features = 8000,
                stop_words = 'english'
            )
        tfidf.fit(data.resumen)
        text = tfidf.transform(data.resumen)
        
        #parametizar n_clusters(dependiendo de lo que el usuario quiera)
        clusters = MiniBatchKMeans(n_clusters=int(codo), init_size=1024, batch_size=2048, random_state=20).fit_predict(text)
        
        #Parametrizar el Size 400, con el contador de los resultados
        def plot_tsne_pca(data, labels):
            max_label = max(labels)
            max_items = np.random.choice(range(data.shape[0]), size=contador, replace=False)
            
            pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
            tsne = TSNE().fit_transform(PCA(n_components=13).fit_transform(data[max_items,:].todense()))
            
            
            idx = np.random.choice(range(pca.shape[0]), size=contador, replace=False)
            label_subset = labels[max_items]
            label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
            
        plot_tsne_pca(text, clusters)
        
        def get_top_keywords(data, clusters, labels, n_terms):
            df = pd.DataFrame(data.todense()).groupby(clusters).mean()
            text = ''
            text2 = ''
            for i,r in df.iterrows():
                clusters = ""
                #eclusters.delete(1.0,END)
                text2 = 'Cluster {}'.format(i) +'\n' + ','.join([labels[t] for t in np.argsort(r)[-n_terms:]]) + '\n'
                text = clusters + text2
                #eclusters.insert(tk.END,text)
                print('\nCluster {}'.format(i))
                print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
                    
        get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)
        clusters = MiniBatchKMeans(n_clusters=codo, init_size=1024, batch_size=2048, random_state=20)
        data['Cluster'] = clusters.fit_predict(text)
        # data.to_csv('out.csv')

        #Procentajes de similitud
        file_docs = []

        #Contexto
        tokens = sent_tokenize(contexto)
        for line in tokens:
            file_docs.append(line)
        #Tokenize words and create dictionary
        gen_docs = [[w.lower() for w in word_tokenize(text)] 
                    for text in file_docs]

        dictionary = gensim.corpora.Dictionary(gen_docs)

        #Create a bag of words
        corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

        #Term Frequency – Inverse Document Frequency(TF-IDF)
        tf_idf = gensim.models.TfidfModel(corpus)

        #Creating similarity measure object
        sims = gensim.similarities.Similarity('../',tf_idf[corpus],
                                                num_features=len(dictionary))

        #Create Query Document
        file2_docs = []

        #Texto 2
        datos = pd.read_csv('resultados.csv', header=0)
        resumen = datos['resumen']

        porcentajes = []
        for line in resumen:
            
            tokens = sent_tokenize(str(line))
            for line in tokens:
                file2_docs.append(line)
            for line in file2_docs:
                query_doc = [w.lower() for w in word_tokenize(line)]
                query_doc_bow = dictionary.doc2bow(query_doc) #update an existing dictionary and create bag of words
            #Avg sims 2
            
            avg_sims = [] # array of averages
            
            for line in file2_docs:
                sum_of_sims = 0
                query_doc = [w.lower() for w in word_tokenize(line)]
                query_doc_bow = dictionary.doc2bow(query_doc)
                query_doc_tf_idf = tf_idf[query_doc_bow]
                sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
                avg = sum_of_sims / len(file_docs)
                avg_sims.append(avg)  
            total_avg = 0
            total_avg = np.sum(avg_sims, dtype=np.float64)
            percentage_of_similarity = round(float(total_avg) * 100)
            porcentajes.append(percentage_of_similarity)
            if percentage_of_similarity >= 100:
                percentage_of_similarity = 100
            avg_sims.clear()
            file2_docs.clear()
        
        
        data['Similitud'] = porcentajes
        #generando archivo de salida
        data.to_csv('out.csv')
        





        #wordcloud
        wordc_list = []
        for i in range(4):
        
            datos = pd.read_csv('out.csv', header=0)
            texto = str(datos[datos.iloc[:,4]==i]['resumen'])
            #texto = str(datos['resumen'])
            
            #Generación de lista de signos de puntuación
            
            punctuation=[]
            for s in string.punctuation:
                punctuation.append(str(s))
            sp_punctuation = ["¿", "¡", "“", "”", "…", ":", "–", "»", "«"]    
            
            punctuation += sp_punctuation
            
            #Listado de palabras que queremos eliminar del texto
            #Es un proceso iterativo por lo que si después vemos que nos siguen quedado "caractéres raros" simplemente venímos aquí y los agregamos
            #Existe librerías y listados de "Stop_words", pero por ahora vamos a dejarlo vacío
            
            #nltk.download('stopwords') #La primera vez debemos descargar las "stopwords"
            
            stop_words = stopwords.words('english') #Listado de palabras a eliminar
            
            stop_words += ["bthe", "bin"] #Añadimos algunos caractéres que hemos encontrado
            
            #Reemplazamos signos de puntuación por "":
            for p in punctuation:
                clean_texto = texto.lower().replace(p,"")
                
            for p in punctuation:
                clean_texto = clean_texto.replace(p,"")
            
            #Eliminamos espacios blancos, saltos de línea, tabuladores, etc    
            #clean_texto = " ".join(clean_texto.split())    
            
            #Reemplazamos stop_words por "":    
            for stop in stop_words:
                clean_texto_list = clean_texto.split()
                clean_texto_list = [i.strip() for i in clean_texto_list]
                try:
                    while stop in clean_texto_list: clean_texto_list.remove(stop)
                except:
                    print("Error")
                    pass
                clean_texto= " ".join(clean_texto_list)
            
                
            lista_texto = clean_texto.split(" ")
            
            palabras = []
            
            #Paso intermedio para eliminar palabras muy cortas (emoticonos,etc) y muy largas (ulr o similar) que se nos hayan pasado:
            
            for palabra in lista_texto:
                if (len(palabra)>=3 and len(palabra)<18):
                    palabras.append(palabra)
                    
            #Generamos un diccionario para contabilizar las palabras:
            
            word_count={}
            
            for palabra in palabras:
                if palabra in word_count.keys():
                    word_count[palabra][0]+=1
                else:
                    word_count[palabra]=[1]
                    
            print(word_count)
            
            #Generamos el DF y lo ordenamos:
            
            df = pd.DataFrame.from_dict(word_count).transpose()
            df.columns=["freq"]
            df.sort_values(["freq"], ascending=False, inplace=True)
            df.head(10)
            
            def plot_bar(data=df, top=5):    
                fig = plt.figure()
                ax = fig.add_axes([0,0,2,1])
                ax.bar(x =df.iloc[:top,:].index, height = df.iloc[:top,0].values)
                
            #Graficamos el TOP 5 palabras por frecuencia
            
            plot_bar(data=df, top=5)
            
            #WordCloud sencillo
            
            word_cloud = WordCloud(height=800, width=800, background_color='white',max_words=150, min_font_size=5, collocation_threshold=10).generate(clean_texto)
            narchivo = "wordc"+str(i)+".png"
            word_cloud.to_file(narchivo) #Guardamos la imagen generada
            
            plt.figure(figsize=(10,8))
            plt.imshow(word_cloud)
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            wordc_list.append(narchivo)
            

        return render(request, "clusters.html", {"wordc_list":wordc_list, "codo":codo})
    else:
        mensaje="no haz introducido ningun valor de busqueda"
    return HttpResponse(mensaje)

def result(request):

    if request.GET["conjunto"]:
        cluster=int(request.GET["conjunto"])
        datos = pd.read_csv('out.csv', header=0)
        
        resumen = datos[datos.iloc[:,4]==cluster]
        resumen = resumen.sort_values(by=['Similitud'], ascending=False)
        print(resumen)
        print(type(resumen))
        # parsing the DataFrame in json format.
        json_records = resumen.reset_index().to_json(orient ='records')
        resultado = []
        resultado = json.loads(json_records)
        context = {'d': resultado}
        print(context)
        return render(request, "resultados.html", context)
    else:
        mensaje="no has introducido ningun valor de busqueda"
    return HttpResponse(mensaje)