<?php
$conexion = new mysqli("localhost", "root", "", "app_mujer");

if ($conexion->connect_error) {
    die("Error de conexión: " . $conexion->connect_error);
}

$correo = $_POST['correo'];
$contrasena = $_POST['contrasena'];

$sql = "SELECT * FROM usuarios WHERE correo='$correo'";
$resultado = $conexion->query($sql);

if ($resultado->num_rows > 0) {
    $usuario = $resultado->fetch_assoc();
    if (password_verify($contrasena, $usuario['contrasena'])) {
        echo "<script>alert('Bienvenida {$usuario['nombre']}'); window.location='menu.html';</script>";
    } else {
        echo "<script>alert('Contraseña incorrecta'); window.location='index.html';</script>";
    }
} else {
    echo "<script>alert('Usuario no encontrado'); window.location='index.html';</script>";
}
?>
