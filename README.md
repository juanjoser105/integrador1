# Instrucciones de instalacion

A continuación se muestran las instrucciones para la instalación y ejecucion

## Verificar la version de Python

Desde la consola ejecutar:
```
python --version
```

# Instalar Python
Instalar desde https://www.python.org/ la versión actual de python



## Proxy para que Google no bloquee nuestra IP por hacer multiples solicitudes a su pagina
Crear una cuenta en [Proxies API](https://app.proxiesapi.com/index.php)
De alli obtienes un KEY, se debe usar en el siguiente comando usando el valor de KEY del sitio anterior y reemplazando la URL de Google Scholar

Desde la consola ejecutar:
```
curl "http://api.proxiesapi.com/?auth_key=XXXXXXXXXXXXXXXXXXXXX&url=https://scholar.google.com/"
```

# Libreria request
Desde la consola ejecutar:
```
pip install requests
```

## Instalaciones requeridas

## Instalar BeautifulSoup framework de webscraping 
Desde la consola ejecutar:
```
pip install beautifulsoup4
```

## Instalar Numpy
Desde la consola ejecutar:
```
pip install numpy
```

## Instalar pandas
Desde la consola ejecutar:
```
pip install pandas
```

## Instalar matplotlib
Desde la consola ejecutar:
```
pip install matplotlib
```

## Instalar gensim
Desde la consola ejecutar:
```
pip install gensim
```

## Instalar nltk
Desde la consola ejecutar:
```
pip install nltk
```

## Instalar PIL para las imagenes
Desde la consola ejecutar:
```
pip install pil
```

## Instalación de wordcloud
Seguir los pasos de este repositorio https://github.com/amueller/word_cloud

## Instalar django
Desde la consola ejecutar:
```
pip install Django
```

## Para que nuestra aplicación haga un buen enrutamiento nos vamos a "settings.py" vamos hasta la linea 58 y reemplazamos la ruta completa hasta la carpeta "/Templates"
```
'DIRS': ["XXXXXXXXXX"],
```


Por ultimo para ejecutar la aplicación se situa la consola en la carpeta donde esté el "manage.py" y allí ejecutar el comando 
```
python manage.py runserver
```
luego nos vamos a nuestro navegador y escribimos en la Barra de direcciones 
```
http://localhost:8000/busquedappl/
```
