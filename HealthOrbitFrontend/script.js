let map;

const countryCoords = {
    "USA": { lat: 37.0902, lng: -95.7129 },
    "Japan": { lat: 36.2048, lng: 138.2529 },
    "Turkey": { lat: 38.9637, lng: 35.2433 },
    "Italy": { lat: 41.8719, lng: 12.5674 },
    "Indonesia": { lat: -0.7893, lng: 113.9213 },
    "Chile": { lat: -35.6751, lng: -71.543 },
    "Nepal": { lat: 28.3949, lng: 84.124 },
    "North Macedonia": { lat: 41.6086, lng: 21.7453 },
};

function initMap() {
    const mapOptions = {
        center: { lat: 20, lng: 0 },
        zoom: 2,
        mapId: '97b2e1425da29a185d30369d'
    };

    map = new google.maps.Map(document.getElementById("map"), mapOptions);

    // Handle country selection zoom
    const countrySelect = document.getElementById("country-select");
    if (countrySelect) {
        countrySelect.addEventListener("change", (e) => {
            const selected = e.target.value;
            if (countryCoords[selected]) {
                map.setCenter(countryCoords[selected]);
                map.setZoom(6);
            }
        });
    }

    fetchEarthquakes();
}

window.initMap = initMap;

function createStyledMarker(color) {
    const div = document.createElement('div');
    div.style.width = '20px';
    div.style.height = '20px';
    div.style.backgroundColor = color;
    div.style.borderRadius = '50%';
    div.style.border = '2px solid white'; // make them distinct
    div.style.boxShadow = '0 0 6px rgba(0,0,0,0.6)';
    return div;
}


function fetchEarthquakes() {
    fetch('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson')
        .then(res => res.json())
        .then(data => {
            const quakes = data.features.sort((a, b) => b.properties.time - a.properties.time);

            // Last quake info
            const latest = quakes[0];
            const latestTime = new Date(latest.properties.time);
            const latestMag = latest.properties.mag.toFixed(1);
            const latestPlace = latest.properties.place;

            const lastInfoDiv = document.getElementById('last-earthquake-info');
            lastInfoDiv.innerHTML = `
                <h2>Last Earthquake in the World</h2>
                <p><strong>Location:</strong> ${latestPlace}</p>
                <p><strong>Magnitude:</strong> ${latestMag}</p>
                <p><strong>Time:</strong> ${latestTime.toLocaleString()}</p>
            `;

            // Summary info
            const totalQuakes = quakes.length;
            const strongest = quakes.reduce((max, eq) => eq.properties.mag > max ? eq.properties.mag : max, 0);
            const summaryDiv = document.getElementById('summary-info');
            summaryDiv.innerHTML = `
                <h2>Summary</h2>
                <p><strong>Total earthquakes this week:</strong> ${totalQuakes}</p>
                <p><strong>Strongest magnitude:</strong> ${strongest.toFixed(1)}</p>
            `;

            // Plot on map
            quakes.forEach(eq => {
                const coords = {
                    lat: eq.geometry.coordinates[1],
                    lng: eq.geometry.coordinates[0],
                };

                const time = new Date(eq.properties.time);
                const now = new Date();
                const diffHours = (now - time) / (1000 * 60 * 60);

                let color;
                if (diffHours < 24) {
                    color = 'darkred';
                } else if (diffHours < 72) {
                    color = 'darkorange';
                } else {
                    color = 'gold';
                }

                const { AdvancedMarkerElement } = google.maps.marker;

                const marker = new AdvancedMarkerElement({
                    map: map,
                    position: coords,
                    title: `${eq.properties.place}\nMagnitude: ${eq.properties.mag}`,
                    content: createStyledMarker(color),
                });

                const infoWindow = new google.maps.InfoWindow({
                    content: `
                        <strong>${eq.properties.place}</strong><br>
                        Mag: ${eq.properties.mag}<br>
                        Time: ${time.toLocaleString()}`
                });

                // ✅ RECOMMENDED: Use 'gmp-click' for AdvancedMarkerElement
                marker.addEventListener('gmp-click', () => {
                    infoWindow.open(map, marker);
                });
            });
        })
        .catch(err => console.error("Failed to fetch earthquakes:", err));
}

const hamburgerBtn = document.getElementById('hamburger');
const slideMenu = document.getElementById('slide-menu');
const profileView = document.getElementById('profile-view');
const alertView = document.getElementById('alert-view');

hamburgerBtn.addEventListener('click', () => {
    slideMenu.classList.toggle('open');
});

// Close menu when clicking outside
document.addEventListener('click', (e) => {
    if (!slideMenu.contains(e.target) && !hamburgerBtn.contains(e.target)) {
        slideMenu.classList.remove('open');
    }
});

// Open Profile view
document.getElementById('menu-profile').addEventListener('click', () => {
    slideMenu.classList.remove('open');
    alertView.classList.remove('open');
    alertView.classList.add('hidden');
    profileView.classList.add('open');
    profileView.classList.remove('hidden');
});

// Open Alert view
document.getElementById('menu-alert').addEventListener('click', () => {
    slideMenu.classList.remove('open');
    profileView.classList.remove('open');
    profileView.classList.add('hidden');
    alertView.classList.add('open');
    alertView.classList.remove('hidden');
});

// Close buttons on side views
document.querySelectorAll('.side-view .close-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        btn.parentElement.classList.remove('open');
        btn.parentElement.classList.add('hidden');
    });
});

// Logout button demo alert
document.getElementById('logout-btn').addEventListener('click', () => {
    alert('Logging out...');
    // Add actual logout logic here
});
const alertsContainer = document.getElementById("alerts");
const alertAudio = document.getElementById("alert-audio");

let knownAlertIds = new Set();  // To track alerts already displayed

async function fetchAlerts() {
    try {
        const response = await fetch("http://127.0.0.1:8000/api/alerts");
        if (!response.ok) throw new Error("Network response not ok");
        const alerts = await response.json();

        // Append only new alerts, don't clear all
        alerts.forEach((alert) => {
            const alertId = `${alert.lat}-${alert.lon}-${alert.timestamp}`;
            if (!knownAlertIds.has(alertId)) {
                knownAlertIds.add(alertId);

                const alertEl = document.createElement("div");
                alertEl.classList.add("alert");
                if (alert.severity === "high") alertEl.classList.add("manual-alert");

                alertEl.textContent = `Severity: ${alert.severity.toUpperCase()} | Location: (${alert.lat.toFixed(4)}, ${alert.lon.toFixed(4)}) | Time: ${new Date(alert.timestamp).toLocaleString()}`;

                alertsContainer.appendChild(alertEl);

                // Play alert sound for new alert
                alertAudio.play();

                // Add pulsing class to new alert and remove from others
                const allAlerts = alertsContainer.querySelectorAll(".alert");
                allAlerts.forEach(a => a.classList.remove("latest-alert"));
                alertEl.classList.add("latest-alert");

                // Scroll to new alert
                alertEl.scrollIntoView({ behavior: "smooth" });
            }
        });
    } catch (err) {
        alertsContainer.textContent = "Error loading alerts.";
        console.error(err);
    }
}


// Fetch alerts every 5 seconds
setInterval(fetchAlerts, 5000);

// Initial fetch
fetchAlerts();
// ... inside fetchAlerts function, after appending all alerts:

// Add pulsing effect to the latest alert
const alertElements = alertsContainer.querySelectorAll(".alert");
if (alertElements.length > 0) {
    alertElements.forEach(el => el.classList.remove("latest-alert"));
    alertElements[alertElements.length - 1].classList.add("latest-alert");
    // Also, you can scroll that alert into view if needed:
    alertElements[alertElements.length - 1].scrollIntoView({ behavior: "smooth" });
}


// Ensure google maps is loaded first (API loads asynchronously)
function initAlertMap(container, lat, lng) {
    return new google.maps.Map(container, {
        center: { lat: lat, lng: lng },
        zoom: 12,
        disableDefaultUI: true,
        gestureHandling: 'none',
    });
}

// Event delegation on alerts container
document.getElementById('alerts').addEventListener('click', function(e) {
    const alertDiv = e.target.closest('.manual-alert');
    if (!alertDiv) return; // click outside an alert, ignore

    // Check if map container already exists inside this alert div
    let mapContainer = alertDiv.querySelector('.alert-map-container');

    if (mapContainer) {
        // Toggle visibility
        if (mapContainer.style.display === 'none') {
            mapContainer.style.display = 'block';
        } else {
            mapContainer.style.display = 'none';
        }
        return;
    }

    // Parse coordinates from text e.g. Location: (41.9981, 21.4254)
    const coordsMatch = alertDiv.textContent.match(/\(([-\d.]+),\s*([-\d.]+)\)/);
    if (!coordsMatch) {
        alert('Could not find coordinates in alert text!');
        return;
    }

    const lat = parseFloat(coordsMatch[1]);
    const lng = parseFloat(coordsMatch[2]);

    // Create a container for the map
    mapContainer = document.createElement('div');
    mapContainer.className = 'alert-map-container';
    mapContainer.style.width = '100%';
    mapContainer.style.height = '150px';
    mapContainer.style.marginTop = '8px';
    alertDiv.appendChild(mapContainer);

    // Initialize map
    const map = initAlertMap(mapContainer, lat, lng);

    // Add marker
    new google.maps.Marker({
        position: { lat, lng },
        map,
        title: `Alert Location (${lat.toFixed(4)}, ${lng.toFixed(4)})`
    });
});
