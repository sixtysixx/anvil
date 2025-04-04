var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 20);
var renderer = new THREE.WebGLRenderer({
  antialias: true
});
var canvas = renderer.domElement;
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(canvas);

// Add lights to the scene
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Soft white light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1); // Bright white light
directionalLight.position.set(5, 5, 5).normalize();
scene.add(directionalLight);

var plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -10);
var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var pointOfIntersection = new THREE.Vector3();
canvas.addEventListener("mousemove", onMouseMove, false);

// Array to hold the cones
const cones = [];

// Function to create a cone
const createcone = (x, y, rotation) => {
    const coneGeometry = new THREE.CylinderGeometry(0.05, 0.175, 5, 64); // radiusTop, radiusBottom, height, radialSegments
    const coneMaterial = new THREE.MeshStandardMaterial({ color: Math.random() * 0xffffff });
    const cone = new THREE.Mesh(coneGeometry, coneMaterial);

    // Position and rotate the cone
    cone.position.set(x, y, 0);
    cone.rotation.z = rotation;

    return cone;
};

// Create and add petals to the scene
const numCones = 48;
for (let i = 0; i < numCones; i++) {
    const angle = (i / numCones) * Math.PI * 2; // Divide the circle into parts
    const x = Math.cos(angle) * 2; // x position
    const y = Math.sin(angle) * 2; // y position
    const cone = createcone(x, y, angle);
    cones.push(cone); // Store the cone in the array
    scene.add(cone); // Add the cone to the scene
}

let rotationSpeed = 0.01; // Speed of rotation

function onMouseMove(event) {
    event.preventDefault();
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = - (event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera)};


    function onClick(event) {
      // update the picking ray with the camera and mouse position
      raycaster.setFromCamera(mouse, camera);
    
      // calculate objects intersecting the picking ray
      const intersects = raycaster.intersectObjects(scene.children);
    
      if (intersects.length > 0) {
        // Get the first intersected object
        const res = intersects.filter(res => res.object.userData.clickable);
        if (res.length > 0) {
          const object = res[0].object;
    
          // Do something with the clicked object
          console.log('You clicked:', object.name);
        }
      }
    }
    


renderer.setAnimationLoop(() => {
    if (resize(renderer)) {
        camera.aspect = canvas.clientWidth / canvas.clientHeight;
        camera.updateProjectionMatrix();
    }

    // Update the position of each cone to make them spin in a circle
    const time = Date.now() * rotationSpeed; // Get the current time
    cones.forEach((cone, index) => {
        const angle = time + (index * (Math.PI * 2) / numCones); // Calculate the angle for each cone
        cone.position.x = Math.cos(angle) * 2; // Update x position
        cone.position.y = Math.sin(angle) * 2; // Update y position
    });

    renderer.render(scene, camera);
});

function resize(renderer) {
    const canvas = renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
        renderer.setSize(width, height, false);
    }
    return needResize;
}

window.addEventListener('mousemove', onMouseMove, false);
window.addEventListener('click', onClick, false);