const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(80, window.innerWidth / window.innerHeight, 0.15, 1000)

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight); // Set the size of the renderer to the window dimensions
renderer.shadowMap.enabled = true; // Enable shadow maps for realistic shadows in the scene
document.body.appendChild(renderer.domElement); // Append the renderer's DOM element (canvas) to the body

// Position the camera so it can view the scene
camera.position.z = 4.5; // Move the camera back along the z-axis to view objects

const sphereGeometry = new THREE.SphereGeometry(.5, 32, 32)
const sphereGeometry2 = new THREE.SphereGeometry(1, 32, 32)
// 0.5: radius, 32: width segments, 32: height segments for a smoother appearance

// Create a Phong material for the sphere, which allows for shiny surfaces
const phongMaterial = new THREE.MeshPhongMaterial({ color: 0x000000, shininess: 255, transparent: true, opacity: 0.85 });
// color: red, shininess: 100 (higher values produce a more reflective surface)
const sphere = new THREE.Mesh(sphereGeometry, phongMaterial); // Combine geometry and material into a mesh
sphere.position.set(-1.5, 0, 0); // Position the sphere to the left of the center
sphere.castShadow = true; // Allow the sphere to cast shadows
scene.add(sphere); // Add the sphere to the scene

// Create a Phong material for the sphere, which allows for shiny surfaces
const phongMaterial2 = new THREE.MeshPhongMaterial({ color: 0xff0000, shininess: 255, transparent: true, opacity: 1 });
// color: red, shininess: 100 (higher values produce a more reflective surface)
const sphere2 = new THREE.Mesh(sphereGeometry2, phongMaterial2); // Combine geometry and material into a mesh
sphere2.position.set(-1.5, 0, -1); // Position the sphere to the left of the center
sphere2.castShadow = true; // Allow the sphere to cast shadows
scene.add(sphere2); // Add the sphere to the scene


// Create a cube geometry with dimensions of 1 unit in all directions
const cubeGeometry = new THREE.BoxGeometry(1, 1, 1); 
const cubeGeometry2 = new THREE.BoxGeometry(1, 1, 1); 
const cubeGeometry3 = new THREE.BoxGeometry(1, 1, 1); 
const cubeGeometry4 = new THREE.BoxGeometry(1, 1, 1); 
// 1: width, height, and depth (uniform size)

// Create a Lambert material for the cube, which gives it a matte finish
const lambertMaterial = new THREE.MeshLambertMaterial({ color: 0x50C878 });
// color: green (no shininess as Lambert material is non-reflective)
const cube = new THREE.Mesh(cubeGeometry, lambertMaterial); // Combine geometry and material into a mesh
cube.position.set(1.5, 0, 0); // Position the cube to the right of the center
cube.castShadow = true; // Allow the cube to cast shadows
scene.add(cube); // Add the cube to the scene

// Create a Lambert material for the cube, which gives it a matte finish
const lambertMaterial2 = new THREE.MeshLambertMaterial({ color: 0x50C878 });
// color: green (no shininess as Lambert material is non-reflective)
const cube2 = new THREE.Mesh(cubeGeometry2, lambertMaterial2); // Combine geometry and material into a mesh
cube2.position.set(1.5, 0, 0); // Position the cube to the right of the center
cube2.castShadow = true; // Allow the cube to cast shadows
scene.add(cube2); // Add the cube to the scene

// Create a Lambert material for the cube, which gives it a matte finish
const lambertMaterial3 = new THREE.MeshLambertMaterial({ color: 0x50C878 });
// color: green (no shininess as Lambert material is non-reflective)
const cube3 = new THREE.Mesh(cubeGeometry3, lambertMaterial2); // Combine geometry and material into a mesh
cube3.position.set(1.5, 0, 0); // Position the cube to the right of the center
cube3.castShadow = true; // Allow the cube to cast shadows
scene.add(cube3); // Add the cube to the scene

// Create a Lambert material for the cube, which gives it a matte finish
const lambertMaterial4 = new THREE.MeshLambertMaterial({ color: 0x50C878 });
// color: green (no shininess as Lambert material is non-reflective)
const cube4 = new THREE.Mesh(cubeGeometry4, lambertMaterial4); // Combine geometry and material into a mesh
cube4.position.set(1.5, 0, 0); // Position the cube to the right of the center
cube4.castShadow = true; // Allow the cube to cast shadows
scene.add(cube4); // Add the cube to the scene

// Add a PointLight to the scene
const pointLight = new THREE.PointLight(0xffffff, 2, 100); 
// color: white, intensity: 1 (brightness), distance: 100 (light will fade beyond this distance)
pointLight.position.set(0, 5, 0); // Position the point light above the objects
pointLight.castShadow = true; // Allow the point light to cast shadows
scene.add(pointLight); // Add the point light to the scene

// Add a SpotLight to the scene
const spotLight = new THREE.SpotLight(0xffffff, 1); 
// color: white, intensity: 1 (brightness)
spotLight.position.set(0, 5, 5); // Position the spotlight above and in front of the objects
spotLight.angle = Math.PI / 8; // Set the angle of the spotlight's light cone (22.5 degrees)
spotLight.penumbra = 0.6; // Soft edges of the light cone for smoother transitions (0 to 1)
spotLight.castShadow = true; // Allow the spotlight to cast shadows
scene.add(spotLight); // Add the spotlight to the scene
/*
// Create an ambient light to provide a base level of illumination
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); 
// color: white, intensity: 0.5 (soft light that affects all objects equally)
scene.add(ambientLight); // Add the ambient light to the scene
*/
// Function to animate the scene
function animate() {
    requestAnimationFrame(animate); // Request the next frame for animation

    // Rotate the objects for visual effect
    sphere.rotation.y += 0.015; // Rotate the sphere around the y-axis (0.015 radians per frame)
    cube.rotation.y += 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube.rotation.x -= 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube2.rotation.y -= 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube2.rotation.x += 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube3.rotation.y -= 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube3.rotation.x -= 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube4.rotation.y += 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)
    cube4.rotation.x += 0.015; // Rotate the cube around the y-axis (0.015 radians per frame)

    // Render the scene from the perspective of the camera
    renderer.render(scene, camera);
}

// Start the animation loop
animate();

// Adjust the camera and renderer when the window is resized
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight; // Update the camera aspect ratio based on new window dimensions
    camera.updateProjectionMatrix(); // Update the projection matrix to account for the new aspect ratio
    renderer.setSize(window.innerWidth, window.innerHeight); // Update the renderer size to match the new window dimensions
});

