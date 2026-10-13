from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
from datetime import datetime
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "clave_secreta_app_mujer"

# Carpeta para guardar evidencia
UPLOAD_FOLDER = "static/evidencia"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------- CONEXIÓN A BASE DE DATOS ---------------------
def conectar_bd():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="app_mujer"
    )

# --------------------- PREGUNTAS DEL TEST ---------------------
PREGUNTAS = [
    "¿Te controla con quién hablas o a dónde vas?",
    "¿Revisa tu teléfono sin permiso?",
    "¿Te dice que nadie más te quiere?",
    "¿Te insulta o te humilla?",
    "¿Te cela demasiado?",
    "¿Te grita cuando se enoja?",
    "¿Te culpa de sus problemas?",
    "¿Te ignora como castigo?",
    "¿Hace comentarios que afectan tu autoestima?",
    "¿Te amenaza con dejarte?",
    "¿Te ha empujado o sujetado fuerte?",
    "¿Te ha golpeado alguna vez?",
    "¿Te obliga a tener relaciones?",
    "¿Te presiona sexualmente?",
    "¿Te ha tocado sin consentimiento?",
    "¿Controla tu dinero?",
    "¿Te impide trabajar o estudiar?",
    "¿Te quita tu dinero?",
    "¿Te limita el acceso a recursos?",
    "¿Amenaza con hacer daño a tus seres queridos?"
]

# --------------------- RUTAS PRINCIPALES ---------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        correo = request.form["correo"]
        contrasena = request.form["contrasena"]

        conexion = conectar_bd()
        cursor = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM usuarias
            WHERE correo = %s AND contrasena = %s
        """, (correo, contrasena))
        usuaria = cursor.fetchone()
        cursor.close()
        conexion.close()

        if usuaria:
            # Guardar datos de usuario en sesión, incluyendo rol
            session["usuaria_id"] = usuaria["id"]
            session["nombre"] = usuaria["nombre"]
            session["apellido"] = usuaria["apellido"]
            session["correo"] = usuaria["correo"]
            session["rol"] = usuaria.get("rol", "usuaria")
            # Si es admin, redirigir al panel admin; si no, al menu
            if session["rol"] == "admin":
                return redirect(url_for("admin_panel"))
            return redirect(url_for("menu"))
        else:
            return render_template("login.html", error="Usuario o contraseña incorrectos")

    return render_template("login.html")

@app.route("/registrarse", methods=["GET", "POST"])
def registro():
    if request.method == "POST":
        nombre = request.form["nombre"]
        apellido = request.form["apellido"]
        correo = request.form["correo"]
        contrasena = request.form["contrasena"]

        conexion = conectar_bd()
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO usuarias (nombre, apellido, correo, contrasena, fecha_registro, rol)
            VALUES (%s, %s, %s, %s, NOW(), 'usuaria')
        """, (nombre, apellido, correo, contrasena))
        conexion.commit()
        cursor.close()
        conexion.close()
        return redirect(url_for("login"))

    return render_template("registro.html")

@app.route("/menu")
def menu():
    if "usuaria_id" not in session:
        return redirect(url_for("login"))
    return render_template("menu.html", nombre=session["nombre"])

# --------------------- TEST ---------------------
@app.route("/realizar_test")
def realizar_test():
    if "usuaria_id" not in session:
        return redirect(url_for("login"))
    return render_template("test.html", preguntas=PREGUNTAS)

@app.route("/procesar_test", methods=["POST"])
def procesar_test():
    if "usuaria_id" not in session:
        return redirect(url_for("login"))

    respuestas = []
    puntaje = 0
    for i in range(len(PREGUNTAS)):
        r = request.form.get(f"respuesta_{i}")
        respuestas.append(r)
        if r == "si":
            puntaje += 1

    nivel = "BAJO" if puntaje <= 6 else "MEDIO" if puntaje <= 13 else "ALTO"

    # Guardar test en la base de datos
    conexion = conectar_bd()
    cursor = conexion.cursor()
    cursor.execute("""
        INSERT INTO test (usuario_id, correo, puntaje, fecha)
        VALUES (%s, %s, %s, NOW())
    """, (session["usuaria_id"], session["correo"], puntaje))
    conexion.commit()
    cursor.close()
    conexion.close()

    # Crear gráficas (pie y bar)
    si_count = respuestas.count("si")
    no_count = respuestas.count("no")
    labels = ["Sí", "No"]
    sizes = [si_count, no_count]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#7a0010", "#f5c242"])
    ax1.axis("equal")
    buf1 = io.BytesIO(); plt.savefig(buf1, format="png"); buf1.seek(0)
    pie_img = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, sizes, color=["#7a0010", "#f5c242"])
    ax2.set_ylabel("Número de respuestas")
    buf2 = io.BytesIO(); plt.savefig(buf2, format="png"); buf2.seek(0)
    bar_img = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)

    return render_template("alerta_enviada.html", mensaje="¡Test enviado correctamente!",
                           pie_img=pie_img, bar_img=bar_img, puntaje=puntaje, nivel=nivel)

# --------------------- HISTORIAL ---------------------
@app.route("/historial")
def historial():
    if "usuaria_id" not in session:
        return redirect(url_for("login"))

    conexion = conectar_bd()
    cursor = conexion.cursor(dictionary=True)
    # Alertas
    cursor.execute("SELECT * FROM alertas WHERE usuario_id = %s ORDER BY fecha DESC", (session["usuaria_id"],))
    alertas = cursor.fetchall()
    # Tests
    cursor.execute("SELECT * FROM test WHERE usuario_id = %s ORDER BY fecha ASC", (session["usuaria_id"],))
    tests = cursor.fetchall()
    cursor.close()
    conexion.close()

    # Gráfica acumulativa
    if tests:
        fechas = [t["fecha"] for t in tests]
        puntajes = [t["puntaje"] for t in tests]
        fig, ax = plt.subplots()
        ax.plot(fechas, puntajes, marker='o', color='#7a0010')
        ax.set_title("Evolución de tus Tests")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Puntaje")
        ax.set_ylim(0, 20)
        fig.autofmt_xdate()
        buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0)
        grafica_acumulativa = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
    else:
        grafica_acumulativa = None

    return render_template("historial.html", alertas=alertas, tests=tests, grafica_acumulativa=grafica_acumulativa)

# --------------------- ALERTAS ---------------------
@app.route("/alertas", methods=["GET", "POST"])
def alertas():
    if "usuaria_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        descripcion = request.form.get("descripcion")
        lat = request.form.get("latitud")
        lng = request.form.get("longitud")
        archivo = request.files.get("archivo")

        nombre_archivo = None
        tipo_archivo = None
        if archivo and archivo.filename != "":
            nombre_archivo = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
            archivo.save(os.path.join(UPLOAD_FOLDER, nombre_archivo))
            tipo_archivo = archivo.content_type

        conexion = conectar_bd()
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO alertas (usuario_id, correo, latitud, longitud, archivo, tipo, descripcion, fecha)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (session["usuaria_id"], session["correo"], lat, lng, nombre_archivo, tipo_archivo, descripcion))
        conexion.commit()
        cursor.close()
        conexion.close()

        return render_template("alerta_enviada.html", mensaje="¡Alerta enviada correctamente!")

    return render_template("alertas.html")

# --------------------- ADMIN (solo accesible a rol admin) ---------------------
@app.route("/admin")
def admin_panel():
    # Solo admin
    if "usuaria_id" not in session:
        return redirect(url_for("login"))
    if session.get("rol") != "admin":
        return "No tienes permisos para acceder a esta página.", 403

    conexion = conectar_bd()
    cursor = conexion.cursor(dictionary=True)

    # Obtener usuarias
    cursor.execute("SELECT * FROM usuarias ORDER BY fecha_registro DESC")
    usuarias = cursor.fetchall()

    # Obtener todos los tests
    cursor.execute("SELECT * FROM test ORDER BY fecha DESC")
    tests = cursor.fetchall()

    # Obtener todas las alertas
    cursor.execute("SELECT * FROM alertas ORDER BY fecha DESC")
    alertas = cursor.fetchall()

    cursor.close()
    conexion.close()

    return render_template("admin_panel.html", usuarias=usuarias, tests=tests, alertas=alertas)

# --------------------- CERRAR SESIÓN ---------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# --------------------- GRÁFICA INDIVIDUAL DE TEST ---------------------
@app.route("/grafico_test/<int:test_id>")
def grafico_test(test_id):
    if "usuaria_id" not in session:
        return redirect(url_for("login"))

    conexion = conectar_bd()
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM test WHERE id = %s AND usuario_id = %s", (test_id, session["usuaria_id"]))
    test = cursor.fetchone()
    cursor.close()
    conexion.close()

    if not test:
        return redirect(url_for("realizar_test"))

    puntaje_si = test["puntaje"]
    puntaje_no = 20 - puntaje_si

    fig1, ax1 = plt.subplots()
    ax1.pie([puntaje_si, puntaje_no], labels=["Sí", "No"], autopct="%1.1f%%",
            startangle=90, colors=["#7a0010", "#f5c242"])
    ax1.axis("equal")
    buf1 = io.BytesIO(); plt.savefig(buf1, format="png"); buf1.seek(0)
    pie_img = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(["Sí", "No"], [puntaje_si, puntaje_no], color=["#7a0010", "#f5c242"])
    ax2.set_ylabel("Número de respuestas")
    buf2 = io.BytesIO(); plt.savefig(buf2, format="png"); buf2.seek(0)
    bar_img = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)

    nivel = "BAJO" if puntaje_si <= 6 else "MEDIO" if puntaje_si <= 13 else "ALTO"

    return render_template("alerta_enviada.html",
                           mensaje="Gráficos individuales del test",
                           pie_img=pie_img,
                           bar_img=bar_img,
                           puntaje=puntaje_si,
                           nivel=nivel)

# --------------------- EJECUTAR SERVIDOR ---------------------
if __name__ == "__main__":
    app.run(debug=True)
