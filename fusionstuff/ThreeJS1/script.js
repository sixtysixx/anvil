// get canvas elemt
const canvas = document.getElementById('threejs-canvas')

// make scene
const scene = new THREE.Scene();

// camera
const camera = new THREE.PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.1, 1000)
camera.position.z = 5

// attach 2 camera

const renderer = new THREE.WebGLRenderer({
    canvas
})
renderer.setSize(window.innerWidth, window.innerHeight)

const geometry = new THREE.BoxGeometry(2, 2, 2);
const material = new THREE.MeshStandardMaterial({
    color: 0xaf7ac5
})

const cube = new THREE.Mesh(geometry, material)
scene.add(cube);

const AmbientLight = new THREE.AmbientLight(0xffffff, .45)
scene.add(AmbientLight)

const DirectionalLight = new THREE.DirectionalLight(0xffffff, 1.2)
DirectionalLight.position.set(5, 5, 5);
scene.add(DirectionalLight)

// Create a cone geometry
const coneGeometry = new THREE.ConeGeometry(1, 2, 32);
const coneMaterial = new THREE.MeshStandardMaterial({
    color: 0xff0000
});
const cone = new THREE.Mesh(coneGeometry, coneMaterial);

// Position the cone
cone.position.x = -3;
scene.add(cone);

// Create cylinder geometry
const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 2, 32);
const cylinderMaterial = new THREE.MeshStandardMaterial({
    color: 0x00ff00
});
const cylinder = new THREE.Mesh(cylinderGeometry, cylinderMaterial);

// Position the cylinder
cylinder.position.x = 3;
scene.add(cylinder);

// Create torus geometry
const torusGeometry = new THREE.TorusGeometry(1, 0.4, 16, 100);
const torusMaterial = new THREE.MeshStandardMaterial({
    color: 0xffff00
});
const torus = new THREE.Mesh(torusGeometry, torusMaterial);

// Position the torus
torus.position.y = -3;
scene.add(torus);

// Create a torus knot geometry
const torusKnotGeometry = new THREE.TorusKnotGeometry(1, 0.4, 100, 16);
const torusKnotMaterial = new THREE.MeshStandardMaterial({
    color: 0x0077ff
});
const torusKnot = new THREE.Mesh(torusKnotGeometry, torusKnotMaterial);

// Position the torus knot
torusKnot.position.x = 0;
torusKnot.position.y = 3; // Slightly above the center
scene.add(torusKnot);


// Change color on click for all shapes
window.addEventListener('click', () => {
    cone.material.color.set(Math.random() * 0xffffff);
    cylinder.material.color.set(Math.random() * 0xffffff);
    torus.material.color.set(Math.random() * 0xffffff);
    torusKnot.material.color.set(Math.random() * 0xffffff);
});


function animate() {
    requestAnimationFrame(animate);

    // Rotate the shapes
    torusKnot.rotation.x += 0.01;
    torusKnot.rotation.y += 0.01;
    cone.rotation.z += 0.01;
    cylinder.rotation.x += 0.01;
    torus.rotation.y += 0.01;
    cube.rotation.z += 0.01;

    // Render the scene
    renderer.render(scene, camera);
}
animate();
