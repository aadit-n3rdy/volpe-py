DOCKER_CMD="${DOCKER_CMD:-podman}"

rm volpe_img.tar
$DOCKER_CMD build -t volpe_grpc_test .
$DOCKER_CMD save -o grpc_test_img.tar volpe_grpc_test
cp volpe_img.tar
