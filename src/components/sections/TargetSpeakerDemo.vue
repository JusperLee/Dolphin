<template>
  <div>
    <el-divider />
    
    <el-row justify="center">
      <h1 class="section-title">Real-World Demo - Target Speaker Separation</h1>
    </el-row>
    
    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="16">
        <div class="demo-description">
          <p>Hover over the target speaker's face to play their separated audio. Red frames indicate target speakers.</p>
        </div>
        
        <!-- Demo Selector -->
        <el-row justify="center" class="demo-selector">
          <el-radio-group v-model="selectedDemo" @change="handleDemoChange">
            <el-radio-button label="demo2">Demo 1</el-radio-button>
            <el-radio-button label="demo1">Demo 2</el-radio-button>
            <el-radio-button label="demo3">Demo 3</el-radio-button>
          </el-radio-group>
        </el-row>
        
        <!-- Video Display Area -->
        <div class="video-demo-container">
          <!-- Mixed Video -->
          <div class="video-section">
            <h3>Mixed Audio Video</h3>
            <div class="video-wrapper main-video-wrapper">
              <video 
                ref="mixVideo"
                :src="mixVideoSrc" 
                controls 
                muted 
                preload="metadata"
                @timeupdate="syncVideos"
                class="main-video"
              >
                您的浏览器不支持视频播放。
              </video>
            </div>
          </div>
          
          <!-- Separated Speaker Videos -->
          <div class="speakers-section">
            <h3>Target Speaker Separation Results</h3>
            <div class="speakers-container">
              <!-- Speaker 1 -->
              <div 
                class="speaker-item"
                @mouseenter="playTargetSpeaker('s1')"
                @mouseleave="stopTargetSpeaker"
              >
                <div class="speaker-label">Speaker 1</div>
                <div class="video-wrapper speaker-video">
                  <!-- Face Image (default state) -->
                  <img 
                    v-show="hoveredSpeaker !== 's1'"
                    :src="s1ImageSrc"
                    alt="Speaker 1 Face"
                    class="face-image"
                  />
                  <!-- Video (hover state) -->
                  <video 
                    ref="s1Video"
                    :src="s1VideoSrc" 
                    :muted="hoveredSpeaker !== 's1'"
                    preload="metadata"
                    class="target-video"
                    v-show="hoveredSpeaker === 's1'"
                  >
                    Your browser does not support video playback.
                  </video>
                  <div class="target-frame" :class="{ active: hoveredSpeaker === 's1' }"></div>
                </div>
              </div>
              
              <!-- Speaker 2 -->
              <div 
                class="speaker-item"
                @mouseenter="playTargetSpeaker('s2')"
                @mouseleave="stopTargetSpeaker"
              >
                <div class="speaker-label">Speaker 2</div>
                <div class="video-wrapper speaker-video">
                  <!-- Face Image (default state) -->
                  <img 
                    v-show="hoveredSpeaker !== 's2'"
                    :src="s2ImageSrc"
                    alt="Speaker 2 Face"
                    class="face-image"
                  />
                  <!-- Video (hover state) -->
                  <video 
                    ref="s2Video"
                    :src="s2VideoSrc" 
                    :muted="hoveredSpeaker !== 's2'"
                    preload="metadata"
                    class="target-video"
                    v-show="hoveredSpeaker === 's2'"
                  >
                    Your browser does not support video playback.
                  </video>
                  <div class="target-frame" :class="{ active: hoveredSpeaker === 's2' }"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Current Playback Status -->
        <div class="playback-status" v-if="hoveredSpeaker">
          <el-alert
            :title="`Now Playing: ${hoveredSpeaker === 's1' ? 'Speaker 1' : 'Speaker 2'} Separated Audio`"
            type="info"
            :closable="false"
            show-icon
          />
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script>
export default {
  name: 'TargetSpeakerDemo',
  data() {
    return {
      selectedDemo: 'demo2',
      hoveredSpeaker: null,
      currentTime: 0
    }
  },
  computed: {
    mixVideoSrc() {
      return `/Dolphin/real-world-demo/${this.selectedDemo}/mix.mp4`
    },
    s1VideoSrc() {
      return `/Dolphin/real-world-demo/${this.selectedDemo}/s1.mp4`
    },
    s2VideoSrc() {
      return `/Dolphin/real-world-demo/${this.selectedDemo}/s2.mp4`
    },
    s1ImageSrc() {
      return `/Dolphin/real-world-demo/${this.selectedDemo}/s1.png`
    },
    s2ImageSrc() {
      return `/Dolphin/real-world-demo/${this.selectedDemo}/s2.png`
    }
  },
  methods: {
    handleDemoChange() {
      // 切换demo时重置状态
      this.hoveredSpeaker = null
      this.pauseAllVideos()
      this.$nextTick(() => {
        this.syncAllVideos()
      })
    },
    
    playTargetSpeaker(speaker) {
      this.hoveredSpeaker = speaker
      
      // 暂停混合视频的音频，但保持视频播放
      if (this.$refs.mixVideo) {
        this.$refs.mixVideo.muted = true
      }
      
      // 播放目标说话人的视频和音频
      const targetVideo = this.$refs[`${speaker}Video`]
      if (targetVideo) {
        targetVideo.currentTime = this.currentTime
        if (this.$refs.mixVideo && !this.$refs.mixVideo.paused) {
          targetVideo.play().catch(e => {
            console.log('Target video play failed:', e)
          })
        }
      }
      
      // 静音其他说话人
      const otherSpeaker = speaker === 's1' ? 's2' : 's1'
      const otherVideo = this.$refs[`${otherSpeaker}Video`]
      if (otherVideo) {
        otherVideo.pause()
      }
    },
    
    stopTargetSpeaker() {
      this.hoveredSpeaker = null
      
      // 恢复混合视频音频
      if (this.$refs.mixVideo) {
        this.$refs.mixVideo.muted = false
      }
      
      // 暂停所有分离视频
      if (this.$refs.s1Video) {
        this.$refs.s1Video.pause()
      }
      if (this.$refs.s2Video) {
        this.$refs.s2Video.pause()
      }
    },
    
    syncVideos() {
      if (!this.$refs.mixVideo) return
      
      this.currentTime = this.$refs.mixVideo.currentTime
      
      // 同步所有视频的播放进度
      if (this.$refs.s1Video) {
        this.$refs.s1Video.currentTime = this.currentTime
      }
      if (this.$refs.s2Video) {
        this.$refs.s2Video.currentTime = this.currentTime
      }
    },
    
    syncAllVideos() {
      // 同步所有视频到主视频的时间
      if (!this.$refs.mixVideo) return
      
      const currentTime = this.$refs.mixVideo.currentTime
      
      if (this.$refs.s1Video) {
        this.$refs.s1Video.currentTime = currentTime
      }
      if (this.$refs.s2Video) {
        this.$refs.s2Video.currentTime = currentTime
      }
    },
    
    pauseAllVideos() {
      if (this.$refs.mixVideo) {
        this.$refs.mixVideo.pause()
      }
      if (this.$refs.s1Video) {
        this.$refs.s1Video.pause()
      }
      if (this.$refs.s2Video) {
        this.$refs.s2Video.pause()
      }
    }
  },
  
  mounted() {
    // 监听主视频的播放/暂停事件，同步到其他视频
    if (this.$refs.mixVideo) {
      this.$refs.mixVideo.addEventListener('play', () => {
        if (this.hoveredSpeaker && this.$refs[`${this.hoveredSpeaker}Video`]) {
          this.$refs[`${this.hoveredSpeaker}Video`].play()
        }
      })
      
      this.$refs.mixVideo.addEventListener('pause', () => {
        if (this.$refs.s1Video) this.$refs.s1Video.pause()
        if (this.$refs.s2Video) this.$refs.s2Video.pause()
      })
    }
  }
}
</script>

<style scoped>
.demo-description {
  text-align: center;
  margin: 20px 0;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 8px;
  color: #606266;
}

.demo-selector {
  margin: 30px 0;
}

.video-demo-container {
  margin-top: 30px;
}

.video-section {
  margin-bottom: 40px;
}

.video-section h3 {
  text-align: center;
  margin-bottom: 20px;
  color: #303133;
}

.speakers-section h3 {
  text-align: center;
  margin-bottom: 20px;
  color: #303133;
}

.video-wrapper {
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.video-wrapper.main-video-wrapper {
  max-width: 50%;
  margin: 0 auto;
}

.main-video {
  width: 100%;
  aspect-ratio: 16 / 9;
  display: block;
}

.speakers-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  max-width: 50%;
  margin: 0 auto;
}

.speaker-item {
  cursor: pointer;
  transition: transform 0.3s ease;
}

.speaker-item:hover {
  transform: translateY(-5px);
}

.speaker-label {
  text-align: center;
  margin-bottom: 10px;
  font-weight: bold;
  color: #409eff;
}

.speaker-video {
  position: relative;
}

.target-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 8px;
  display: block;
}

.face-image {
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
  border-radius: 8px;
  transition: opacity 0.3s ease;
}

.target-frame {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border: 3px solid transparent;
  border-radius: 8px;
  transition: border-color 0.3s ease;
  pointer-events: none;
}

.target-frame.active {
  border-color: #ff4757;
  box-shadow: 0 0 15px rgba(255, 71, 87, 0.5);
}

.playback-status {
  margin-top: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .speakers-container {
    grid-template-columns: 1fr;
    gap: 15px;
  }
  
  .demo-description {
    font-size: 14px;
    padding: 10px;
  }
}
</style>