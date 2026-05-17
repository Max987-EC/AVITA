// 負責管線模式的圖層與視圖渲染

// ==========================================
// 1. 圖層管理與排序邏輯 (Layer Management)
// ==========================================
function addNewLayerFromSelect() {
    const select = document.getElementById('newLayerSelect');
    const type = select.value;
    const name = select.options[select.selectedIndex].text;
    
    const layerId = 'layer_' + Date.now();
    layerStack.push({
        id: layerId,
        type: type,
        name: name,
        params: {}
    });
    
    selectLayer(layerId);
    renderTopLayers();
    debouncedProcessImage();
}

function removeLayer(id, event) {
    event.stopPropagation();
    layerStack = layerStack.filter(layer => layer.id !== id);
    
    if (activeLayerId === id) {
        activeLayerId = layerStack.length > 0 ? layerStack[layerStack.length - 1].id : null;
    }
    
    renderTopLayers();
    if (activeLayerId) selectLayer(activeLayerId);
    else paramPanel.style.display = 'none';
    debouncedProcessImage();
}

function moveLayer(index, direction, event) {
    event.stopPropagation();
    if (direction === -1 && index > 0) {
        [layerStack[index - 1], layerStack[index]] = [layerStack[index], layerStack[index - 1]];
    } else if (direction === 1 && index < layerStack.length - 1) {
        [layerStack[index + 1], layerStack[index]] = [layerStack[index], layerStack[index + 1]];
    }
    renderTopLayers();
    debouncedProcessImage();
}

function selectLayer(id) {
    if (currentMode !== 'stack') return;
    
    saveCurrentParamsToLayer();
    activeLayerId = id;
    renderTopLayers();
    
    const layer = layerStack.find(l => l.id === id);
    if (layer) {
        showParamGroupForType(layer.type);
        loadParamsIntoDOM(layer);
        
        const index = layerStack.findIndex(l => l.id === id);
        if (currentPipelineData && currentPipelineData[index]) {
            renderNodeView(index);
        }
    }
}

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
// 2. 管線視圖渲染函數 (Pipeline Views)
// ==========================================
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

function renderNodeView(nodeIndex) {
    const nodeData = currentPipelineData[nodeIndex];
    if (!nodeData) return;
    
    document.getElementById('originalImage').src = "data:image/jpeg;base64," + nodeData.input_img;
    document.getElementById('processedImage').src = "data:image/jpeg;base64," + nodeData.output_img;
    
    document.getElementById('originalHistogram').src = "data:image/png;base64," + nodeData.original_histogram;
    document.getElementById('processedHistogram').src = "data:image/png;base64," + nodeData.processed_histogram;
    document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + nodeData.original_spectrum;
    document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + nodeData.processed_spectrum;
    
    document.getElementById('histogramSection').style.display = 'block';
    document.getElementById('spectrumSection').style.display = 'block';
    
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
