# Dashboard de publicaciones de la CEPAL
## Marco Espinoza
## Agosto, 2024

![logo-lab](Image.png)

# Configuración en ambiente remoto

## 1. Log-in en servidor de desarrollo

```bash
# conexión SSH
ssh myserveruser@sgo-ub22-sea-d1
# ingresan su password
myserverpassword
```
## 2. En la carpeta src

**NOTA:** Para clonar el repositorio se puede utilizar el **SSH keys** o el **HTTPS**. 
Además, este paso se realiza únicamente **una vez**. 

```bash
git clone https://gitlab.cepal.org/prospective-lab/dashboard-publicaciones.git

# ingresan usuario y contraseña de GitLab
mygitlabuser
mygitlabpassword
```
Posteriormente, deben abrir la carpeta del repositorio con el siguiente comando:

```bash
cd dashboard-publicaciones
```

## 3. Ambiente

### 3.1 Crear ambiente
Inicialmente, debemos crear el ambiente. Esto se realiza una **única vez**.

```bash
python3 -m venv myenv
```

### 3.2 Activación del ambiente

```bash
source myenv/bin/activate
```

### 3.3 Instalación de dependencias 

```bash
pip3 install -r requirements.txt 
```

## 4. Ejecución del archivo de Streamlit

 ```bash
streamlit run streamlit_app.py
```

## 5. Persistencia de la app sin desconexión.

```bash
nohup streamlit run streamlit_app.py >data.out 2>&1 &
```

```bash
tail -f data.out 
```


# Proceso de desarrollo

Nuevas características agregadas al dashboard deben primero ser desarrolladas de forma **local**. Una vez que se esté de acuerdo con los cambios realizados, se ejecuta lo siguiente: 

```bash
# Agregamos al stage los archivos que queremos guardar
git add .
```

```bash
# Importante que el mensaje que agreguemos al commit, sea lo suficientemente robusto para 
# que el equipo pueda entender los cambios que se realizan en el archivo.
git commit -m "agregamos un mensaje con el commit"
```

```bash
# Hacemos el push a la rama de desarrollo que sea necesaria.
git push
```

## Buenas prácticas

1. Es importante que cuando se desarrolle una nueva características se trabaje en una **rama**/*branch* distinto y luego mediante un **pull request** se unifican los cambios con la rama principal. Esto con el fin de evitar que el código **funcional** se caiga. 

2. Cuando los cambios hayan sido unificados de forma exitosa en la rama principal. En el servidor de desarrollo ejecutamos: 

```bash
# Nos ubicamos en el branch/rama 
git pull
```