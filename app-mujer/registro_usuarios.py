from flask import Flask, render_template, request, redirect, url_for, jsonify
from violencia_db.conexion_db import obtener_conexion
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/evidencias'

# Página principal
@app.route('/')
def login():
    return render_template('login.html')

# Validación de login
@app.route('/login', methods=['POST'])
def login_usuario():
    correo = request.form['correo']
    contrasena = request.form['contrasena']
    conexion = obtener_conexion()
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM usuarias WHERE correo=%s AND contrasena=%s", (correo, contrasena))
    usuario = cursor.fetchone()
    cursor.close()
    conexion.close()
    if usuario:
        return redirect(url_for('usuario_panel', id=usuario['id']))
    else:
        return "Usuario o contraseña incorrectos"

# Panel del usuario
@app.route('/usuario/<int:id>')
def usuario_panel(id):
    return render_template('usuario_panel.html', id=id)

# Registro de nuevas usuarias
@app.route('/registrar', methods=['POST'])
def registrar():
    nombre = request.form['nombre']
    apellido = request.form['apellido']
    correo = request.form['correo']
    contrasena = request.form['contrasena']
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("INSERT INTO usuarias (nombre, apellido, correo, contrasena) VALUES (%s,%s,%s,%s)", 
                   (nombre, apellido, correo, contrasena))
    conexion.commit()
    cursor.close()
    conexion.close()
    return redirect('/')

# Insertar test
@app.route('/insertar_test', methods=['POST'])
def insertar_test():
    usuario_id = request.form['usuario_id']
    correo = request.form['correo']
    puntaje = request.form['puntaje']
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("INSERT INTO test (usuario_id, correo, puntaje) VALUES (%s,%s,%s)", 
                   (usuario_id, correo, puntaje))
    conexion.commit()
    cursor.close()
    conexion.close()
    return jsonify({'mensaje': 'Test guardado correctamente'})

# Insertar historial
@app.route('/insertar_historial', methods=['POST'])
def insertar_historial():
    usuario_id = request.form['usuario_id']
    descripcion = request.form['descripcion']
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("INSERT INTO historial (usuario_id, descripcion) VALUES (%s,%s)", 
                   (usuario_id, descripcion))
    conexion.commit()
    cursor.close()
    conexion.close()
    return jsonify({'mensaje': 'Historial registrado correctamente'})

# Insertar alerta (mapa + evidencia)
@app.route('/insertar_alerta', methods=['POST'])
def insertar_alerta():
    usuario_id = request.form['usuario_id']
    correo = request.form['correo']
    latitud = request.form['latitud']
    longitud = request.form['longitud']
    tipo = request.form['tipo']
    descripcion = request.form['descripcion']
    archivo = request.files['archivo']
    nombre_archivo = None
    if archivo:
        nombre_archivo = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
        ruta = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
        archivo.save(ruta)
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("""INSERT INTO alertas 
                      (usuario_id, correo, latitud, longitud, archivo, tipo, descripcion) 
                      VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                   (usuario_id, correo, latitud, longitud, nombre_archivo, tipo, descripcion))
    conexion.commit()
    cursor.close()
    conexion.close()
    return jsonify({'mensaje': 'Alerta enviada correctamente'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
