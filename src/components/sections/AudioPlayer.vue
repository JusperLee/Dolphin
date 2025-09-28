<template>
    <el-divider />
  <div class="audio-player-section">
    <h2 class="section-title">Audio Separation Results</h2>
    
    <!-- 音频文件选择器 -->
    <div class="audio-selector">
      <label for="audio-select">Select Audio File:</label>
      <select id="audio-select" v-model="selectedAudio" @change="onAudioChange">
        <option value="">Please select an audio file</option>
        <option v-for="audio in audioFiles" :key="audio" :value="audio">
          {{ formatAudioName(audio) }}
        </option>
      </select>
    </div>

    <!-- 音频播放器和频谱图展示 -->
    <div class="audio-content">
      <div class="audio-grid">
        <!-- 混合音频 -->
        <div class="audio-item">
          <h3>Mixture</h3>
          <div class="spectrogram-container">
            <img :src="getSpectrogramPath('mixture', selectedAudio)" :alt="`Mixture spectrogram`" />
          </div>
          <audio controls :src="getAudioPath('mixture', selectedAudio)">
            Your browser does not support audio playback.
          </audio>
        </div>

        <!-- Ground Truth -->
        <div class="audio-item">
          <h3>Ground Truth</h3>
          <div class="spectrogram-container">
            <img :src="getSpectrogramPath('gt', selectedAudio)" :alt="`Ground truth spectrogram`" />
          </div>
          <audio controls :src="getAudioPath('gt', selectedAudio)">
            Your browser does not support audio playback.
          </audio>
        </div>

        <!-- Dolphin 结果 -->
        <div class="audio-item">
          <h3>Dolphin (Ours)</h3>
          <div class="spectrogram-container">
            <img :src="getSpectrogramPath('Dolphin', selectedAudio)" :alt="`Dolphin separation result spectrogram`" />
          </div>
          <audio controls :src="getAudioPath('Dolphin', selectedAudio)">
            Your browser does not support audio playback.
          </audio>
        </div>

        <!-- AV-Mossformer2 结果 -->
        <div class="audio-item">
          <h3>AV-Mossformer2</h3>
          <div class="spectrogram-container">
            <img :src="getSpectrogramPath('AV-Mossformer2', selectedAudio)" :alt="`AV-Mossformer2 separation result spectrogram`" />
          </div>
          <audio controls :src="getAudioPath('AV-Mossformer2', selectedAudio)">
            Your browser does not support audio playback.
          </audio>
        </div>

        <!-- IIANet 结果 -->
        <div class="audio-item">
          <h3>IIANet</h3>
          <div class="spectrogram-container">
            <img :src="getSpectrogramPath('IIANet', selectedAudio)" :alt="`IIANet separation result spectrogram`" />
          </div>
          <audio controls :src="getAudioPath('IIANet', selectedAudio)">
            Your browser does not support audio playback.
          </audio>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AudioPlayer',
  data() {
    return {
      selectedAudio: '6339077953587836416_00025_1.048_6338273935710083967_00003_-1.048',
      audioFiles: [
        '6339077953587836416_00025_1.048_6338273935710083967_00003_-1.048',
        '6355861826787125225_00011_0.88796_6360685934053997353_00007_-0.88796',
        '6365141533126941637_00051_1.5209_6369259547770281088_00021_-1.5209',
        '6368494184598133861_00017_4.2493_6363654615448994597_00013_-4.2493',
        '6375480378401815497_00001_2.2431_6368494184598133861_00017_-2.2431',
        '6382189546814897943_00019_0.74266_6338273935710083967_00003_-0.74266'
      ]
    }
  },
  methods: {
    formatAudioName(filename) {
      // 简化显示名称
      const parts = filename.split('_');
      return `Audio Sample ${parts[1] || '1'}`;
    },
    getAudioPath(method, filename) {
      return `/Dolphin/output_audio/${method}/${filename}.wav`;
    },
    getSpectrogramPath(method, filename) {
      return `/Dolphin/output_audio_mag/${method}/${filename}.png`;
    },
    onAudioChange() {
      // 停止所有正在播放的音频
      const audioElements = this.$el.querySelectorAll('audio');
      audioElements.forEach(audio => {
        audio.pause();
        audio.currentTime = 0;
      });
    }
  }
}
</script>

<style scoped>
.audio-player-section {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.section-title {
  text-align: center;
  color: #333;
  margin-bottom: 2rem;
  font-size: 2rem;
  font-weight: bold;
}

.audio-selector {
  margin-bottom: 2rem;
  text-align: center;
}

.audio-selector label {
  font-weight: bold;
  margin-right: 1rem;
  color: #555;
}

.audio-selector select {
  padding: 0.5rem 1rem;
  border: 2px solid #ddd;
  border-radius: 8px;
  font-size: 1rem;
  background-color: white;
  cursor: pointer;
  transition: border-color 0.3s ease;
}

.audio-selector select:hover {
  border-color: #007bff;
}

.audio-content {
  margin-top: 2rem;
}

.audio-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.audio-item {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.audio-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

.audio-item h3 {
  margin: 0 0 1rem 0;
  color: #333;
  font-size: 1.2rem;
  text-align: center;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.spectrogram-container {
  margin-bottom: 1rem;
  text-align: center;
  background: white;
  border-radius: 8px;
  padding: 1rem;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.spectrogram-container img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

audio {
  width: 100%;
  margin-top: 1rem;
  border-radius: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .audio-grid {
    grid-template-columns: 1fr;
  }
  
  .audio-player-section {
    padding: 1rem;
  }
  
  .section-title {
    font-size: 1.5rem;
  }
}
</style>