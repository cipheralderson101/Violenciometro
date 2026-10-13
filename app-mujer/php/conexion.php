<?php
$conexion = new mysqli("localhost", "root", "", "app_mujer");

if ($conexion->connect_error) {
    die("Error de conexión: " . $conexion->connect_error);
}
?>
