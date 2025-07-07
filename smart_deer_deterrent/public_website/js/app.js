// Configuration
const AZURE_STORAGE_URL = 'https://deerdetections.blob.core.windows.net/deer-detections';
const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes
const DETECTIONS_INDEX_URL = `${AZURE_STORAGE_URL}/api/detections.json`;

// State
let allDetections = [];
let filteredDetections = [];
let refreshTimer = null;

// DOM Elements
let gallery, loading, noResults, totalDetections, dateRange, dateFilter, timeFilter;
let confidenceFilter, confidenceValue, modalVideo, modalTitle, modalDetails;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    gallery = document.getElementById('gallery');
    loading = document.getElementById('loading');
    noResults = document.getElementById('no-results');
    totalDetections = document.getElementById('total-detections');
    dateRange = document.getElementById('date-range');
    dateFilter = document.getElementById('date-filter');
    timeFilter = document.getElementById('time-filter');
    confidenceFilter = document.getElementById('confidence-filter');
    confidenceValue = document.getElementById('confidence-value');
    modalVideo = document.getElementById('modal-video');
    modalTitle = document.getElementById('modal-title');
    modalDetails = document.getElementById('modal-details');
    
    initializeFilters();
    loadDetections();
    startAutoRefresh();
});

// Filter initialization
function initializeFilters() {
    // Don't set a default date - show all
    dateFilter.value = '';
    
    // Update confidence value display
    confidenceFilter.addEventListener('input', (e) => {
        confidenceValue.textContent = `${e.target.value}%`;
        applyFilters();
    });
    
    // Add filter event listeners
    dateFilter.addEventListener('change', applyFilters);
    timeFilter.addEventListener('change', applyFilters);
}

// Load detections from Azure
async function loadDetections() {
    showLoading(true);
    
    try {
        // Determine API URL based on where we're running
        let apiUrl = '/api/detections.json';
        if (window.location.hostname.includes('blob.core.windows.net')) {
            apiUrl = `${AZURE_STORAGE_URL}/api/detections.json`;
        }
        
        const response = await fetch(apiUrl);
        const data = await response.json();
        allDetections = data.detections;
        
        updateStats();
        applyFilters();
        
    } catch (error) {
        console.error('Error loading detections:', error);
        showError('Failed to load detections. Please try again later.');
    }
}

// Generate mock data for testing
async function generateMockData() {
    const detections = [];
    const dates = ['2025-06-30', '2025-07-01'];
    const times = ['06:15:30', '08:45:12', '11:30:45', '14:20:18', '17:50:33', '20:15:42'];
    
    dates.forEach(date => {
        times.forEach(time => {
            const timestamp = `${date}T${time}`;
            const confidence = Math.random() * 0.5 + 0.5; // 0.5 to 1.0
            
            detections.push({
                id: `detection_${date.replace(/-/g, '')}_${time.replace(/:/g, '')}`,
                timestamp: timestamp,
                date: date,
                time: time,
                confidence: confidence,
                labels: confidence > 0.8 ? ['deer'] : ['deer', 'animal'],
                video_url: `${AZURE_STORAGE_URL}/videos/${date}/detection_${date.replace(/-/g, '')}_${time.replace(/:/g, '')}.mp4`,
                thumbnail_url: `${AZURE_STORAGE_URL}/thumbnails/${date}/detection_${date.replace(/-/g, '')}_${time.replace(/:/g, '')}.jpg`,
                metadata: {
                    duration: 30,
                    fps: 10,
                    resolution: '1920x1080'
                }
            });
        });
    });
    
    return { detections: detections.sort((a, b) => b.timestamp.localeCompare(a.timestamp)) };
}

// Apply filters to detections
function applyFilters() {
    const selectedDate = dateFilter.value;
    const selectedTime = timeFilter.value;
    const minConfidence = confidenceFilter.value / 100;
    
    filteredDetections = allDetections.filter(detection => {
        // Date filter
        if (selectedDate && detection.date !== selectedDate) {
            return false;
        }
        
        // Time filter
        if (selectedTime) {
            const hour = parseInt(detection.time.split(':')[0]);
            switch (selectedTime) {
                case 'morning':
                    if (hour < 6 || hour >= 12) return false;
                    break;
                case 'afternoon':
                    if (hour < 12 || hour >= 18) return false;
                    break;
                case 'evening':
                    if (hour < 18 || hour >= 24) return false;
                    break;
                case 'night':
                    if (hour >= 6) return false;
                    break;
            }
        }
        
        // Confidence filter
        if (detection.confidence < minConfidence) {
            return false;
        }
        
        return true;
    });
    
    renderGallery();
}

// Render the gallery
function renderGallery() {
    showLoading(false);
    
    if (filteredDetections.length === 0) {
        gallery.innerHTML = '';
        noResults.style.display = 'block';
        return;
    }
    
    noResults.style.display = 'none';
    
    gallery.innerHTML = filteredDetections.map((detection, index) => `
        <div class="detection-card" data-index="${index}" onclick="playVideoByIndex(${index})">
            <img class="detection-thumbnail" 
                 src="${detection.thumbnail_url}" 
                 alt="Detection at ${detection.time}"
                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 320 200%22><rect fill=%22%23ddd%22 width=%22320%22 height=%22200%22/><text x=%22160%22 y=%22100%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%2216%22>No Thumbnail</text></svg>'">
            <div class="detection-info">
                <div class="detection-time">${formatTime(detection.time)}</div>
                <div class="detection-date">${formatDate(detection.date)}</div>
                <div class="detection-meta">
                    <div class="detection-labels">
                        ${detection.labels.map(label => `<span class="label-badge">${label}</span>`).join('')}
                    </div>
                    <span class="confidence-badge ${getConfidenceClass(detection.confidence)}">
                        ${Math.round(detection.confidence * 100)}%
                    </span>
                </div>
            </div>
        </div>
    `).join('');
}

// Play video by index
function playVideoByIndex(index) {
    const detection = filteredDetections[index];
    const videoUrl = detection.video_url;  // Use direct URL, not SAS
    playVideo(videoUrl, detection);
}

// Play video in modal
function playVideo(videoUrl, detection) {
    const modal = document.getElementById('video-modal');
    
    console.log('Playing video:', videoUrl);
    
    modalTitle.textContent = `Detection - ${formatDate(detection.date)} ${formatTime(detection.time)}`;
    modalVideo.src = videoUrl;
    
    modalDetails.innerHTML = `
        <div><strong>Confidence:</strong> ${Math.round(detection.confidence * 100)}%</div>
        <div><strong>Labels:</strong> ${detection.labels.join(', ')}</div>
        <div><strong>Duration:</strong> ${detection.metadata.duration}s</div>
        ${detection.metadata.resolution ? `<div><strong>Resolution:</strong> ${detection.metadata.resolution}</div>` : ''}
    `;
    
    modal.style.display = 'block';
    
    // Add error handling
    modalVideo.onerror = function(e) {
        console.error('Video failed to load:', e);
        modalDetails.innerHTML += '<div style="color: red; margin-top: 10px;">Error loading video. The URL may have expired.</div>';
    };
    
    modalVideo.play().catch(err => {
        console.error('Play failed:', err);
    });
}

// Close video modal
function closeVideo() {
    const modal = document.getElementById('video-modal');
    modal.style.display = 'none';
    modalVideo.pause();
    modalVideo.src = '';
}

// Update statistics
function updateStats() {
    totalDetections.textContent = allDetections.length;
    
    if (allDetections.length > 0) {
        const dates = allDetections.map(d => d.date);
        const minDate = dates[dates.length - 1];
        const maxDate = dates[0];
        dateRange.textContent = minDate === maxDate ? formatDate(minDate) : `${formatDate(minDate)} - ${formatDate(maxDate)}`;
    } else {
        dateRange.textContent = 'No detections';
    }
}

// Helper functions
function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
    gallery.style.display = show ? 'none' : 'grid';
}

function showError(message) {
    gallery.innerHTML = `<div class="error-message">${message}</div>`;
    showLoading(false);
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
}

function formatTime(timeStr) {
    const [hours, minutes] = timeStr.split(':');
    const hour = parseInt(hours);
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour % 12 || 12;
    return `${displayHour}:${minutes} ${ampm}`;
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
}

// Auto-refresh
function startAutoRefresh() {
    refreshTimer = setInterval(loadDetections, REFRESH_INTERVAL);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('video-modal');
    if (event.target === modal) {
        closeVideo();
    }
}