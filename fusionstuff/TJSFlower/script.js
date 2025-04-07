// Create the scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

// Create a renderer
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Position the camera
camera.position.z = 5;

const createPetal = (x, y, rotation) => {
    const petalGeometry = new THREE.CylinderGeometry(0, .175, 2.5, 64); // radiusTop, radiusBottom, height, radialSegments
    const petalMaterial = new THREE.MeshStandardMaterial({ color: Math.random() * 0xffffff });
    const petal = new THREE.Mesh(petalGeometry, petalMaterial);

    // Position and rotate the petal
    petal.position.set(0, 0, 0);
    petal.rotation.z = rotation;

    return petal;
};

// Create and add petals to the scene
for (let i = 0; i < 16; i++) {
    const angle = (i / 16) * Math.PI * 2; // Divide the circle into 8 parts
    const x = Math.cos(angle) * 2; // x position
    const y = Math.sin(angle) * 2; // y position
    const petal = createPetal(x, y, angle);
    scene.add(petal);
}

// Create the center of the flower
const centerGeometry = new THREE.SphereGeometry(0.5, 32, 32);
const centerMaterial = new THREE.MeshStandardMaterial({ color: 0xffd700 }); // Gold color
const center = new THREE.Mesh(centerGeometry, centerMaterial);
scene.add(center);

// Add Ambient Light
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

// Add Directional Light
const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(5, 5, 5);
scene.add(directionalLight);

function animate() {
    requestAnimationFrame(animate);

    // Rotate the flower
    scene.rotation.y += 0.01;

    // Render the scene
    renderer.render(scene, camera);
}

// Start the animation
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});