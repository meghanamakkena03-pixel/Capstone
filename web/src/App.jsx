
import { useEffect, useRef, useState } from 'react';
import {
  Waves,
  ArrowRight,
  Upload,
  ImagePlus,
  Trash2,
  CheckCircle2,
  LoaderCircle,
  AlertCircle,
  Sparkles,
  ScanSearch,
  Layers3,
  Eye,
  ShieldCheck,
  Leaf,
  FileImage,
  RefreshCw,
  Clock3,
} from 'lucide-react';
import './App.css';

const API_URL = 'https://capstone-backend-dtau.onrender.com';
const MAX_FILE_SIZE = 15 * 1024 * 1024;

const workflow = [
  {
    number: '01',
    title: 'Introduction',
    description: 'Explore an underwater image with AI-assisted analysis.',
    icon: Waves,
    status: 'ready',
  },
  {
    number: '02',
    title: 'Image Enhancement',
    description: 'Improve underwater visibility, brightness and contrast.',
    icon: Sparkles,
    status: 'ready',
  },
  {
    number: '03',
    title: 'Pollution Detection',
    description: 'Locate underwater waste and identify detected trash classes.',
    icon: ScanSearch,
    status: 'ready',
  },
  {
    number: '04',
    title: 'Image Segmentation',
    description: 'Highlight pollution regions at the pixel level.',
    icon: Layers3,
    status: 'pending',
  },
  {
    number: '05',
    title: 'Explainable AI',
    description: 'Visualize model attention using a Grad-CAM heatmap.',
    icon: Eye,
    status: 'pending',
  },
];

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState('idle');
  const [backendResult, setBackendResult] = useState(null);
  const [imageError, setImageError] = useState('');
  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    return () => {
      if (selectedImage?.preview) {
        URL.revokeObjectURL(selectedImage.preview);
      }
    };
  }, [selectedImage]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setImageError('');

    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setImageError('Please choose a valid image file.');
      event.target.value = '';
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setImageError('Your image must be smaller than 15 MB.');
      event.target.value = '';
      return;
    }

    setSelectedImage({
      file,
      preview: URL.createObjectURL(file),
    });

    setBackendResult(null);
    setAnalysisStatus('idle');
  };

  const removeImage = () => {
    setSelectedImage(null);
    setBackendResult(null);
    setAnalysisStatus('idle');
    setImageError('');

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      setImageError('Choose an underwater image before starting analysis.');
      return;
    }

    setAnalysisStatus('loading');
    setImageError('');
    setBackendResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage.file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail || `Analysis failed with status ${response.status}.`
        );
      }

      setBackendResult(result);
      setAnalysisStatus('success');

      window.setTimeout(() => {
        resultsRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 100);
    } catch (error) {
      setImageError(
        error.message === 'Failed to fetch'
          ? 'Cannot connect to Aqua Scope. Please try again later.'
          : error.message || 'Something went wrong during analysis.'
      );
      setAnalysisStatus('error');
    }
  };

  const trashDetections = backendResult?.trash_detections || [];
  const nonTrashObjects = backendResult?.non_trash_objects || [];

  // Stage 04: read segmentation results returned by the backend.
  const segmentationImage =
    backendResult?.segmentation_image ||
    backendResult?.segmented_image ||
    null;

  const segmentationStatus = backendResult?.segmentation_status;
  const segmentationError = backendResult?.segmentation_error;

  const segmentationSegments = Array.isArray(backendResult?.segments)
    ? backendResult.segments
    : [];

  return (
    <div className="app-shell" id="home">
      <header className="navbar">
        <a className="brand" href="#home" aria-label="Aqua Scope home">
          <span className="brand-icon">
            <Waves size={23} />
          </span>
          <span className="brand-text">
            <strong>
              Aqua<span>Scope</span>
            </strong>
            <small>UNDERWATER INTELLIGENCE</small>
          </span>
        </a>

        <nav className="nav-links" aria-label="Main navigation">
          <a href="#home">Home</a>
          <a href="#enhancement">Enhancement</a>
          <a href="#detection">Detection</a>
          <a href="#segmentation">Segmentation</a>
          <a href="#explainability">Explainability</a>
        </nav>

        <a href="#analyze" className="nav-button">
          Analyze an image <ArrowRight size={16} />
        </a>
      </header>

      <main>
        {/* 01 — INTRODUCTION */}
        <section className="hero-section">
          <div className="hero-copy">
            <div className="eyebrow">
              <span className="eyebrow-dot" />
              AI-ASSISTED OCEAN RESEARCH
            </div>

            <h1>
              A clearer view
              <br />
              of the <span>underwater world.</span>
            </h1>

            <p className="hero-description">
              Explore underwater images through image enhancement and
              pollution detection. Understand what is visible beneath the
              surface, one step at a time.
            </p>

            <div className="hero-actions">
              <a className="primary-button" href="#analyze">
                Start analysis <ArrowRight size={17} />
              </a>
              <a className="text-button" href="#workflow">
                Explore the workflow
              </a>
            </div>

            <div className="hero-note">
              <ShieldCheck size={17} />
              <span>Built for underwater image research</span>
            </div>
          </div>

          <div className="hero-art" aria-hidden="true">
            <div className="hero-art-top">
              <span className="art-dot" />
              AQUA SCOPE / IMAGE INTELLIGENCE
            </div>

            <div className="ocean-illustration">
              <div className="ocean-ring ring-one" />
              <div className="ocean-ring ring-two" />
              <div className="ocean-center">
                <Waves size={88} strokeWidth={1.1} />
              </div>
              <span className="ocean-bubble bubble-one" />
              <span className="ocean-bubble bubble-two" />
              <span className="ocean-bubble bubble-three" />
            </div>

            <div className="art-bottom">
              <span>
                <span className="art-bottom-dot" />
                COMPUTER VISION
              </span>
              <span>01 — 05</span>
            </div>
          </div>
        </section>

        {/* WORKFLOW OVERVIEW */}
        <section className="workflow-section section-wrap" id="workflow">
          <div className="section-heading">
            <div>
              <span className="section-kicker">THE PROCESS</span>
              <h2>Five steps. One clear workflow.</h2>
              <p>
                Each stage has its own purpose, from improving visibility
                to explaining model predictions.
              </p>
            </div>
            <span className="workflow-count">05 STAGES</span>
          </div>

          <div className="workflow-list">
            {workflow.map((stage) => {
              const Icon = stage.icon;
              const isReady = stage.status === 'ready';

              return (
                <a
                  className="workflow-item"
                  href={
                    stage.number === '01'
                      ? '#home'
                      : stage.number === '02'
                        ? '#enhancement'
                        : stage.number === '03'
                          ? '#detection'
                          : stage.number === '04'
                            ? '#segmentation'
                            : '#explainability'
                  }
                  key={stage.number}
                >
                  <span className="workflow-number">{stage.number}</span>
                  <span className="workflow-icon">
                    <Icon size={21} />
                  </span>
                  <span className="workflow-copy">
                    <strong>{stage.title}</strong>
                    <small>{stage.description}</small>
                  </span>
                  <span
                    className={`stage-status ${isReady ? 'status-ready' : 'status-pending'}`}
                  >
                    {isReady ? 'Model available' : 'Coming next'}
                  </span>
                  <ArrowRight className="workflow-arrow" size={17} />
                </a>
              );
            })}
          </div>
        </section>

        {/* IMAGE UPLOAD */}
        <section className="upload-section section-wrap" id="analyze">
          <div className="section-heading">
            <div>
              <span className="section-kicker">GET STARTED</span>
              <h2>Bring your image into focus.</h2>
              <p>
                Upload an underwater image to run enhancement and pollution
                detection.
              </p>
            </div>
          </div>

          <div className="upload-layout">
            <div className="upload-panel">
              <input
                ref={fileInputRef}
                className="hidden-file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                aria-label="Choose an underwater image"
              />

              {!selectedImage ? (
                <div className="empty-upload">
                  <div className="upload-icon">
                    <ImagePlus size={27} />
                  </div>
                  <h3>Select an underwater image</h3>
                  <p>
                    Choose a JPG, PNG, or another supported image format.
                  </p>
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload size={17} /> Choose image
                  </button>
                  <span className="upload-limit">Maximum file size: 15 MB</span>
                </div>
              ) : (
                <div className="selected-file">
                  <div className="selected-file-heading">
                    <div className="file-heading-icon">
                      <FileImage size={20} />
                    </div>
                    <div className="file-heading-copy">
                      <strong>Image selected</strong>
                      <small>Ready for AI analysis</small>
                    </div>
                    <button
                      className="icon-button"
                      type="button"
                      onClick={removeImage}
                      aria-label="Remove selected image"
                      title="Remove image"
                    >
                      <Trash2 size={17} />
                    </button>
                  </div>

                  <img
                    className="selected-preview"
                    src={selectedImage.preview}
                    alt="Selected underwater scene"
                  />

                  <div className="file-details">
                    <span className="file-name">{selectedImage.file.name}</span>
                    <span>
                      {(selectedImage.file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>

                  <button
                    className="change-image-button"
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <RefreshCw size={15} /> Choose a different image
                  </button>
                </div>
              )}

              {imageError && (
                <div className="message-error" role="alert">
                  <AlertCircle size={18} />
                  <span>{imageError}</span>
                </div>
              )}

              {selectedImage && (
                <button
                  className="primary-button analyze-button"
                  type="button"
                  onClick={handleAnalyze}
                  disabled={analysisStatus === 'loading'}
                >
                  {analysisStatus === 'loading' ? (
                    <>
                      <LoaderCircle className="spin-icon" size={18} />
                      Analyzing image...
                    </>
                  ) : (
                    <>
                      Analyze image <ArrowRight size={17} />
                    </>
                  )}
                </button>
              )}

              {analysisStatus === 'loading' && (
                <div className="loading-note" role="status">
                  <Clock3 size={16} />
                  <span>
                    Enhancement and detection may take a little time on CPU.
                  </span>
                </div>
              )}
            </div>

            <aside className="analysis-info">
              <div className="info-icon">
                <ShieldCheck size={22} />
              </div>
              <h3>What happens to your image?</h3>
              <p>
                Your image is sent to the local Python backend for processing.
                The current analysis runs enhancement and trash detection.
              </p>

              <div className="info-divider" />

              <div className="info-step">
                <span className="info-step-number">01</span>
                <div>
                  <strong>Enhance</strong>
                  <small>NU²Net improves image visibility.</small>
                </div>
              </div>

              <div className="info-step">
                <span className="info-step-number">02</span>
                <div>
                  <strong>Detect</strong>
                  <small>YOLO searches for known trash classes.</small>
                </div>
              </div>

              <div className="info-disclaimer">
                <AlertCircle size={15} />
                Segmentation and Grad-CAM are not connected yet.
              </div>
            </aside>
          </div>
        </section>

        {/* RESULTS */}
        {analysisStatus === 'success' && backendResult && (
          <section className="results-section section-wrap" ref={resultsRef}>
            <div className="section-heading">
              <div>
                <span className="section-kicker">ANALYSIS RESULTS</span>
                <h2>Your image, step by step.</h2>
                <p>
                  Results are shown in the same order as the Aqua Scope
                  processing pipeline.
                </p>
              </div>
              <span className="success-pill">
                <CheckCircle2 size={15} /> Analysis complete
              </span>
            </div>

            {/* 02 — ENHANCEMENT */}
            <section className="result-stage" id="enhancement">
              <div className="result-stage-heading">
                <span className="result-stage-number">02</span>
                <div>
                  <span className="section-kicker">IMAGE ENHANCEMENT</span>
                  <h3>A clearer underwater image</h3>
                  <p>
                    Compare the uploaded image with the output generated by
                    NU²Net.
                  </p>
                </div>
              </div>

              <div className="image-comparison">
                <ImageResult
                  title="Original image"
                  subtitle="Uploaded input"
                  src={backendResult.original_image}
                />
                <ImageResult
                  title="Enhanced image"
                  subtitle="NU²Net output"
                  src={backendResult.enhanced_image}
                />
              </div>

              <div className="result-caption">
                <Sparkles size={16} />
                Enhancement is performed independently of trash detection.
              </div>
            </section>

            {/* 03 — POLLUTION DETECTION */}
            <section className="result-stage" id="detection">
              <div className="result-stage-heading">
                <span className="result-stage-number">03</span>
                <div>
                  <span className="section-kicker">POLLUTION DETECTION</span>
                  <h3>Objects identified by the model</h3>
                  <p>
                    YOLO detections are made on the original image. The same
                    bounding boxes are also displayed over the enhanced image.
                  </p>
                </div>
              </div>

              <div className="image-comparison">
                <ImageResult
                  title="Trash detection"
                  subtitle="Original image with bounding boxes"
                  src={backendResult.original_detection_image}
                />
                <ImageResult
                  title="Detection on enhanced view"
                  subtitle="Same original-image detections"
                  src={backendResult.enhanced_detection_image}
                />
              </div>

              <div className="detection-summary">
                <div className="summary-stat">
                  <span className="summary-label">Trash detections</span>
                  <strong>{trashDetections.length}</strong>
                  <small>
                    {trashDetections.length === 1
                      ? 'object identified'
                      : 'objects identified'}
                  </small>
                </div>

                <div className="summary-stat">
                  <span className="summary-label">Other object classes</span>
                  <strong>{nonTrashObjects.length}</strong>
                  <small>distinct classes detected</small>
                </div>

                <div className="summary-stat summary-message">
                  <span className="summary-label">Model result</span>
                  <strong>
                    {trashDetections.length > 0
                      ? 'Trash detected'
                      : 'No trash detected'}
                  </strong>
                  <small>
                    This describes the model output, not proof that the image
                    is pollution-free.
                  </small>
                </div>
              </div>

              <div className="detection-list-panel">
                <div className="detection-list-heading">
                  <h4>Detected trash objects</h4>
                  <span>{trashDetections.length} results</span>
                </div>

                {trashDetections.length > 0 ? (
                  <div className="detection-list">
                    {trashDetections.map((item, index) => (
                      <div
                        className="detection-row"
                        key={`${item.class_name}-${index}`}
                      >
                        <span className="detection-index">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <div className="detection-object">
                          <strong>{formatClassName(item.class_name)}</strong>
                          <small>
                            Bounding box: {item.box?.join(', ') || 'Available'}
                          </small>
                        </div>
                        <span className="confidence-value">
                          {(item.confidence * 100).toFixed(1)}%
                          <small>confidence</small>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-detections">
                    <ScanSearch size={22} />
                    <strong>No trash objects were detected.</strong>
                    <p>
                      This does not guarantee the image is pollution-free.
                      Detection depends on image quality and model performance.
                    </p>
                  </div>
                )}
              </div>

              {nonTrashObjects.length > 0 && (
                <div className="other-objects">
                  <strong>Other detected classes</strong>
                  <div className="object-tags">
                    {nonTrashObjects.map((name) => (
                      <span className="object-tag" key={name}>
                        {formatClassName(name)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* 04 — SEGMENTATION */}
            <section className="future-stage" id="segmentation">
              <div className="future-icon">
                <Layers3 size={23} />
              </div>
              <div className="future-copy">
                <span className="section-kicker">STAGE 04</span>
                <h3>Image segmentation</h3>

                {segmentationImage ? (
                  <>
                    <p>
                      The segmentation model highlighted the pollution regions
                      in the uploaded underwater image.
                    </p>

                    <div className="segmentation-result">
                      <ImageResult
                        title="Pixel-level trash segmentation"
                        subtitle="Segmentation model output"
                        src={segmentationImage}
                      />
                    </div>

                    {segmentationSegments.length > 0 && (
                      <div className="segmentation-details">
                        <strong>Segment details</strong>

                        <div className="object-tags">
                          {segmentationSegments.map((segment, index) => {
                            const name =
                              typeof segment === 'string'
                                ? segment
                                : segment.class_name ||
                                  segment.label ||
                                  segment.name ||
                                  segment.class ||
                                  `Segment ${index + 1}`;

                            return (
                              <span
                                className="object-tag"
                                key={`${name}-${index}`}
                              >
                                {formatClassName(name)}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <span className="future-status">
                      Segmentation complete
                    </span>
                  </>
                ) : (
                  <>
                    <p>
                      {segmentationStatus === 'skipped_no_trash_detected'
                        ? 'Segmentation was skipped because the detection stage did not identify any trash objects.'
                        : segmentationStatus === 'failed'
                          ? segmentationError ||
                            'The backend reported an error while running segmentation.'
                          : 'The backend did not return a segmentation image. Check the segmentation model and the response from the /analyze endpoint.'}
                    </p>

                    {segmentationStatus && (
                      <p className="segmentation-status-detail">
                        Status: {segmentationStatus}
                      </p>
                    )}

                    <span className="future-status">
                      {segmentationStatus === 'skipped_no_trash_detected'
                        ? 'Skipped — no trash detected'
                        : segmentationStatus === 'failed'
                          ? 'Segmentation failed'
                          : 'No segmentation output returned'}
                    </span>
                  </>
                )}
              </div>
            </section>

            {/* 05 — EXPLAINABILITY */}
            
{/* 05 — EXPLAINABILITY */}
<section className="future-stage" id="explainability">
  <div className="future-icon">
    <Eye size={23} />
  </div>

  <div className="future-copy">
    <span className="section-kicker">STAGE 05</span>
    <h3>Explainable AI</h3>

    {backendResult?.explainability_heatmap ? (
      <>
        <p>
          The visualization highlights the trash objects detected by the
          model. Labels show the detected class and confidence score.
        </p>

        <div className="segmentation-result">
          <ImageResult
            title="Detection-based explanation"
            subtitle="Highlighted trash objects and confidence scores"
            src={backendResult.explainability_heatmap}
          />
        </div>

        {trashDetections.length > 0 ? (
          <div className="segmentation-details">
            <strong>Objects highlighted in the explanation</strong>

            <div className="object-tags">
              {trashDetections.map((item, index) => (
                <span
                  className="object-tag"
                  key={`${item.class_name}-${index}`}
                >
                  {formatClassName(item.class_name)}
                  {typeof item.confidence === 'number'
                    ? ` · ${(item.confidence * 100).toFixed(1)}%`
                    : ''}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <p>No trash objects were detected to highlight.</p>
        )}

        <span className="future-status">
          {backendResult.explainability_status === 'completed'
            ? 'Explanation generated'
            : 'Explanation image available'}
        </span>

        <p className="segmentation-status-detail">
          This is a detection-based visualization, not a true Grad-CAM
          activation heatmap.
        </p>
      </>
    ) : (
      <>
        <p>
          {backendResult?.explainability_status === 'no_trash_detected'
            ? 'No trash objects were detected, so there are no trash detections to highlight.'
            : backendResult?.explainability_status === 'failed'
              ? backendResult?.explainability_error ||
                'The backend could not generate the explanation.'
              : 'The backend did not return an explanation image. Check the explainability status and backend response.'}
        </p>

        {backendResult?.explainability_status && (
          <p className="segmentation-status-detail">
            Status: {backendResult.explainability_status}
          </p>
        )}

        <span className="future-status">
          {backendResult?.explainability_status === 'no_trash_detected'
            ? 'No trash to explain'
            : backendResult?.explainability_status === 'failed'
              ? 'Explanation failed'
              : 'No explanation output returned'}
        </span>
      </>
    )}
  </div>
</section>

          </section>
        )}

        {/* FUTURE FEATURES WHEN THERE ARE NO RESULTS */}
        {analysisStatus !== 'success' && (
          <section className="future-overview section-wrap">
            <div className="future-overview-heading">
              <div>
                <span className="section-kicker">COMING IN LATER STAGES</span>
                <h2>More ways to understand the ocean.</h2>
              </div>
              <Leaf size={27} />
            </div>

            <div className="future-overview-grid">
              <article className="future-overview-card" id="segmentation">
                <Layers3 size={22} />
                <span>04</span>
                <h3>Image segmentation</h3>
                <p>
                  Highlight individual pollution regions after a segmentation
                  model is integrated.
                </p>
                <small>Model needed</small>
              </article>

              <article className="future-overview-card" id="explainability">
                <Eye size={22} />
                <span>05</span>
                <h3>Explainable AI</h3>
                <p>
                  Generate Grad-CAM visualizations after the explanation
                  implementation is configured.
                </p>
                <small>Implementation needed</small>
              </article>
            </div>
          </section>
        )}

        <section className="about-section section-wrap" id="about">
          <div className="about-icon">
            <Waves size={25} />
          </div>
          <div>
            <span className="section-kicker">ABOUT AQUA SCOPE</span>
            <h2>Technology for a clearer view beneath the surface.</h2>
            <p>
              Aqua Scope is an underwater image analysis project that combines
              computer vision and image enhancement to support the study of
              underwater environments and pollution.
            </p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand footer-brand" href="#home">
          <span className="brand-icon">
            <Waves size={20} />
          </span>
          <span className="brand-text">
            <strong>
              Aqua<span>Scope</span>
            </strong>
            <small>UNDERWATER INTELLIGENCE</small>
          </span>
        </a>
        <p>Underwater image analysis · AI-assisted research</p>
        <a href="#home" className="back-to-top">
          Back to top ↑
        </a>
      </footer>
    </div>
  );
}

function ImageResult({ title, subtitle, src }) {
  return (
    <article className="image-result-card">
      <div className="image-result-heading">
        <div>
          <h4>{title}</h4>
          <p>{subtitle}</p>
        </div>
        <ImagePlus size={18} />
      </div>
      {src ? (
        <img src={src} alt={title} className="result-image" />
      ) : (
        <div className="missing-image">
          <AlertCircle size={20} />
          <span>Image result is unavailable.</span>
        </div>
      )}
    </article>
  );
}

function formatClassName(name = '') {
  return name
    .replace(/^trash_/, '')
    .replace(/^animal_/, '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export default App;
