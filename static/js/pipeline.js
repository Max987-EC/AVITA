// 檢測管線管理邏輯 (Pipeline & Layer Management)
// 負責管線模式下的算子堆疊操作、順序調整及多層級視圖的渲染。

// ==========================================
// 1. 圖層堆疊操作 (Layer Stack Operations)
// ==========================================

/**
 * 從下拉選單新增一個新算子圖層至堆疊末尾。
 */
function addNewLayerFromSelect() {
    const select = document.getElementById('newLayerSelect');
    const type = select.value;
    const name = select.options[select.selectedIndex].text;
    
    // 產生唯一 ID 並初始化參數
    const layerId = 'layer_' + Date.now();
    layerStack.push({
        id: layerId,
        type: type,
        name: name,
        params: {}
    });
    
    selectLayer(layerId); // 自動選中新圖層
    renderTopLayers();    // 更新左側清單 UI
    debouncedProcessImage();
}

/**
 * 移除指定 ID 的圖層。
 * @param {string} id - 圖層唯一識別碼
 * @param {Event} event - 點擊事件 (用於防止事件冒泡)
 */
function removeLayer(id, event) {
    event.stopPropagation();
    layerStack = layerStack.filter(layer => layer.id !== id);
    
    // 若被刪除的是當前選中層，則將選中項移至最後一層
    if (activeLayerId === id) {
        activeLayerId = layerStack.length > 0 ? layerStack[layerStack.length - 1].id : null;
    }
    
    renderTopLayers();
    if (activeLayerId) selectLayer(activeLayerId);
    else paramPanel.style.display = 'none';
    
    debouncedProcessImage();
}

/**
 * 調整圖層在堆疊中的順序 (上移/下移)。
 * @param {number} index - 目前索引
 * @param {number} direction - 位移方向 (-1 上移, 1 下移)
 */
function moveLayer(index, direction, event) {
    event.stopPropagation();
    if (direction === -1 && index > 0) {
        // 交換位置
        [layerStack[index - 1], layerStack[index]] = [layerStack[index], layerStack[index - 1]];
    } else if (direction === 1 && index < layerStack.length - 1) {
        [layerStack[index + 1], layerStack[index]] = [layerStack[index], layerStack[index + 1]];
    }
    renderTopLayers();
    debouncedProcessImage();
}

/**
 * 選中特定圖層，並載入其參數至面板。
 * @param {string} id - 圖層 ID
 */
function selectLayer(id) {
    if (currentMode !== 'stack') return;
    
    saveCurrentParamsToLayer(); // 儲存舊層參數
    activeLayerId = id;
    renderTopLayers();
    
    const layer = layerStack.find(l => l.id === id);
    if (layer) {
        showParamGroupForType(layer.type);
        loadParamsIntoDOM(layer);
        
        // 同步更新視圖為該節點的輸入/輸出
        const index = layerStack.findIndex(l => l.id === id);
        if (currentPipelineData && currentPipelineData[index]) {
            renderNodeView(index);
        }
    }
}

/**
 * 渲染左側管線算子清單 UI。
 */
function renderTopLayers() {
    const list = document.getElementById('topLayerList');
    if (!list) return;
    list.innerHTML = '';

    layerStack.forEach((layer, index) => {
        const item = document.createElement('div');
        item.className = `top-layer-item ${layer.id === activeLayerId ? 'active' : ''}`;
        
        item.onclick = () => selectLayer(layer.id);
        item.style.cursor = 'pointer';
        
        item.innerHTML = `
            <div class="top-layer-name" style="flex:1;">${index + 1}. ${layer.name}</div>
            <div class="top-layer-actions">
                <button onclick="moveLayer(${index}, -1, event)" ${index === 0 ? 'disabled style="opacity:0.3"' : ''}>▲</button>
                <button onclick="moveLayer(${index}, 1, event)" ${index === layerStack.length - 1 ? 'disabled style="opacity:0.3"' : ''}>▼</button>
                <button class="btn-del" onclick="removeLayer('${layer.id}', event)">✖</button>
            </div>
        `;
        list.appendChild(item);
    });
}

// ==========================================
// 2. 管線視圖渲染 (Pipeline View Rendering)
// ==========================================

/**
 * 全局視圖：顯示整個管線的最起始原圖與最終輸出結果。
 */
function renderGlobalView() {
    activeLayerId = null;
    renderTopLayers(); 
    paramPanel.style.display = 'none'; 
    
    document.getElementById('originalImage').src = globalInputUrl;
    document.getElementById('processedImage').src = globalOutputUrl;
    
    document.getElementById('originalHistogram').src = "data:image/png;base64," + globalOriginalHistogram;
    document.getElementById('processedHistogram').src = "data:image/png;base64," + globalProcessedHistogram;
    document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + globalOriginalSpectrum;
    document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + globalProcessedSpectrum;
    
    document.getElementById('histogramSection').style.display = 'block';
    document.getElementById('spectrumSection').style.display = 'block';
    
    // 在 Steps 區域列出所有節點的處理結果快照
    const stepsContainer = document.getElementById('stepsContainer');
    const stepsGrid = document.getElementById('stepsGrid');
    stepsGrid.innerHTML = ''; 
    
    if (currentPipelineData.length > 0) {
        stepsContainer.style.display = 'block';
        currentPipelineData.forEach((node, index) => {
            const stepBox = document.createElement('div');
            stepBox.className = 'step-box';
            stepBox.innerHTML = `
                <h5>[ Node ${index + 1} ] ${node.operation_name}</h5>
                <img src="data:image/jpeg;base64,${node.output_img}" alt="Node Output">
            `;
            stepBox.style.cursor = 'pointer';
            stepBox.onclick = () => selectLayer(layerStack[index].id);
            stepsGrid.appendChild(stepBox);
        });
    } else {
        stepsContainer.style.display = 'none';
    }
}

/**
 * 節點視圖：詳細觀察管線中特定步驟的「輸入」與「輸出」。
 * @param {number} nodeIndex - 節點索引
 */
function renderNodeView(nodeIndex) {
    const nodeData = currentPipelineData[nodeIndex];
    if (!nodeData) return;
    
    // 將該節點的輸入影像設為左側，輸出影像設為右側
    document.getElementById('originalImage').src = "data:image/jpeg;base64," + nodeData.input_img;
    document.getElementById('processedImage').src = "data:image/jpeg;base64," + nodeData.output_img;
    
    document.getElementById('originalHistogram').src = "data:image/png;base64," + nodeData.original_histogram;
    document.getElementById('processedHistogram').src = "data:image/png;base64," + nodeData.processed_histogram;
    document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + nodeData.original_spectrum;
    document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + nodeData.processed_spectrum;
    
    document.getElementById('histogramSection').style.display = 'block';
    document.getElementById('spectrumSection').style.display = 'block';
    
    // 在 Steps 區域顯示該算子內部的中間處理步驟 (若有)
    const stepsContainer = document.getElementById('stepsContainer');
    const stepsGrid = document.getElementById('stepsGrid');
    stepsGrid.innerHTML = ''; 
    
    if (nodeData.steps && nodeData.steps.length > 0) {
        stepsContainer.style.display = 'block';
        nodeData.steps.forEach(step => {
            const stepBox = document.createElement('div');
            stepBox.className = 'step-box';
            stepBox.innerHTML = `
                <h5>[ ${step.name} ]</h5>
                <img src="data:image/jpeg;base64,${step.image}" alt="${step.name}">
            `;
            stepsGrid.appendChild(stepBox);
        });
    } else {
        stepsContainer.style.display = 'none';
    }
}
