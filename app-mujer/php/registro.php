<?php
$conexion = new mysqli("localhost", "root", "", "app_mujer");

if ($conexion->connect_error) {
    die("Error de conexión: " . $conexion->connect_error);
}

$nombre = $_POST['nombre'];
$apellido = $_POST['apellido'];
$correo = $_POST['correo'];
$contrasena = password_hash($_POST['contrasena'], PASSWORD_DEFAULT);

$sql = "INSERT INTO usuarios (nombre, apellido, correo, contrasena) VALUES ('$nombre', '$apellido', '$correo', '$contrasena')";

if ($conexion->query($sql) === TRUE) {
    echo "<script>alert('Registro exitoso. Ahora puedes iniciar sesión.'); window.location='index.html';</script>";
} else {
    echo "<script>alert('Error al registrar: " . $conexion->error . "'); window.location='registro.html';</script>";
}
?>
