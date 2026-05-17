// 負責相機、放大鏡、預設圖片等獨立 UI

// ==========================================
// 1. 預設圖片載入邏輯 (Preset Images)
// ==========================================
async function loadPresetImage(filename) {
    const loading = document.getElementById('loadingIndicator');
    if(loading) loading.classList.add('show');
    
    try {
        const response = await fetch(`/static/samples/${filename}`);
        if (!response.ok) throw new Error("找不到圖片檔案");
        
        const blob = await response.blob();
        const file = new File([blob], filename, { type: blob.type });
        
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        const imageInput = document.getElementById('imageInput');
        imageInput.files = dataTransfer.files;
        
        imageInput.dispatchEvent(new Event('change'));
        
    } catch (error) {
        console.error("載入預設圖片失敗:", error);
        alert(`無法載入 ${filename}！請確認圖片已放在 static/samples/ 資料夾中。`);
    } finally {
        if(loading) loading.classList.remove('show');
    }
}

// ==========================================
// 2. 圖片放大鏡功能 (Image Modal Zoom & Pan)
// ==========================================
let currentScale = 1;
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;

function openImageModal(imgSrc, captionText) {
    const imageModal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImage");
    const caption = document.getElementById("modalCaption");
    
    if (!imageModal || !modalImg) return;

    imageModal.removeAttribute("onclick");
    imageModal.style.display = "block";
    modalImg.src = imgSrc;
    caption.innerHTML = captionText || "影像預覽";
    
    currentScale = 1;
    translateX = 0;
    translateY = 0;
    modalImg.style.transform = `translate(0px, 0px) scale(1)`;
    
    setTimeout(() => { modalImg.style.cursor = 'grab'; }, 300);
}

function closeImageModal() {
    const imageModal = document.getElementById("imageModal");
    if (imageModal) imageModal.style.display = "none";
}

document.addEventListener('click', function(e) {
    if (e.target.tagName === 'IMG' && (e.target.closest('.image-box') || e.target.closest('.step-box'))) {
        if (e.target.src && e.target.src.length > 10) {
            openImageModal(e.target.src, e.target.alt);
        }
    } else if (e.target.id === 'imageModal' || e.target.classList.contains('close-modal')) {
        closeImageModal();
    }
});

document.addEventListener('wheel', function(e) {
    if (e.target.id === 'modalImage') {
        e.preventDefault(); 
        const zoomSpeed = 0.15;
        currentScale += (e.deltaY < 0) ? zoomSpeed : -zoomSpeed;
        currentScale = Math.min(Math.max(0.5, currentScale), 15); 
        e.target.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentScale})`;
    }
}, { passive: false });

document.addEventListener('mousedown', function(e) {
    if (e.target.id === 'modalImage') {
        e.preventDefault();
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
        e.target.style.cursor = 'grabbing';
    }
});

document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    const modalImg = document.getElementById("modalImage");
    if (modalImg) {
        e.preventDefault();
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentScale})`;
    }
});

document.addEventListener('mouseup', function() {
    if (isDragging) {
        isDragging = false;
        const modalImg = document.getElementById("modalImage");
        if (modalImg) modalImg.style.cursor = 'grab';
    }
});

document.addEventListener('keydown', function(e) {
    if (e.key === "Escape") closeImageModal();
});

// ==========================================
// 3. 實時相機擷取功能 (WebRTC Camera)
// ==========================================
let currentStream = null;

async function openCamera() {
    document.getElementById('cameraModal').style.display = 'block';
    await getCameras();
}

function closeCamera() {
    document.getElementById('cameraModal').style.display = 'none';
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
}

async function getCameras() {
    try {
        const initialStream = await navigator.mediaDevices.getUserMedia({ video: true });
        initialStream.getTracks().forEach(track => track.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        const cameraSelect = document.getElementById('cameraSelect');
        cameraSelect.innerHTML = ''; 
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `攝影機 ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        if (videoDevices.length > 0) startStream(videoDevices[0].deviceId);
        cameraSelect.onchange = () => startStream(cameraSelect.value);

    } catch (err) {
        console.error("無法存取攝影機:", err);
        alert("無法存取攝影機，請確認瀏覽器權限已開啟！");
    }
}

async function startStream(deviceId) {
    if (currentStream) currentStream.getTracks().forEach(track => track.stop());
    
    const constraints = {
        video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1920 }, height: { ideal: 1080 } }
    };

    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('cameraVideo');
        video.srcObject = currentStream;
    } catch (err) {
        console.error("啟動影像流失敗:", err);
    }
}

document.addEventListener('click', function(e) {
    const captureBtn = e.target.closest('#captureBtn');
    if (!captureBtn) return;

    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    if (!video || !canvas) return;

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        const fileInput = document.querySelector('input[type="file"]');
        
        if (fileInput) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        } else {
            alert("找不到圖片上傳欄位，請確認左側面板結構。");
        }
        closeCamera();
    }, 'image/jpeg', 0.95);
});
