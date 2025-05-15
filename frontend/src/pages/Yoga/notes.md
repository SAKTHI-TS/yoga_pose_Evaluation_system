import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import React, { useRef, useState, useEffect } from 'react';
import backend from '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import { count } from '../../utils/music';
import Instructions from '../../components/Instrctions/Instructions';
import './Yoga.css';
import DropDown from '../../components/DropDown/DropDown';
import { poseImages } from '../../utils/pose_images';
import { POINTS, keypointConnections } from '../../utils/data';
import { drawPoint, drawSegment } from '../../utils/helper';

let skeletonColor = 'rgb(255,255,255)';
let poseList = [
  'Tree', 'Chair', 'Cobra', 'Warrior', 'Dog',
  'Shoulderstand', 'Traingle'
];

// Global variables (outside of React's state system)
let interval;
let trainingInterval;
let samplingInterval;
let confidenceValues = []; // Store confidence values
let currentConfidenceValue = 0; // Current confidence value
let isCurrentlyTraining = false; // Global training flag
let flag = false; // For pose detection

const CLASS_NO = {
  Chair: 0,
  Cobra: 1,
  Dog: 2,
  No_Pose: 3,
  Shoulderstand: 4,
  Traingle: 5,
  Tree: 6,
  Warrior: 7,
};

function Yoga() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const classifierRef = useRef(null);

  const [startingTime, setStartingTime] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [poseTime, setPoseTime] = useState(0);
  const [bestPerform, setBestPerform] = useState(0);
  const [currentPose, setCurrentPose] = useState('Tree');
  const [isStartPose, setIsStartPose] = useState(false);
  const [countdown, setCountdown] = useState(20);
  const [score, setScore] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [samples, setSamples] = useState(0);
  const [currentConfidence, setCurrentConfidence] = useState(0);

  useEffect(() => {
    const timeDiff = (currentTime - startingTime) / 1000;
    if (flag) {
      setPoseTime(timeDiff);
    }
    if ((currentTime - startingTime) / 1000 > bestPerform) {
      setBestPerform(timeDiff);
    }
  }, [currentTime]);

  useEffect(() => {
    setCurrentTime(0);
    setPoseTime(0);
    setBestPerform(0);
  }, [currentPose]);

  // Cleanup effect for intervals
  useEffect(() => {
    return () => {
      cleanupAllIntervals();
    };
  }, []);

  function cleanupAllIntervals() {
    if (interval) clearInterval(interval);
    if (trainingInterval) clearInterval(trainingInterval);
    if (samplingInterval) clearInterval(samplingInterval);
  }

  function get_center_point(landmarks, left_bodypart, right_bodypart) {
    let left = tf.gather(landmarks, left_bodypart, 1);
    let right = tf.gather(landmarks, right_bodypart, 1);
    const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5));
    return center;
  }

  function get_pose_size(landmarks, torso_size_multiplier = 2.5) {
    let hips_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP);
    let shoulders_center = get_center_point(landmarks, POINTS.LEFT_SHOULDER, POINTS.RIGHT_SHOULDER);
    let torso_size = tf.norm(tf.sub(shoulders_center, hips_center));
    let pose_center_new = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP);
    pose_center_new = tf.expandDims(pose_center_new, 1);

    pose_center_new = tf.broadcastTo(pose_center_new, [1, 17, 2]);
    let d = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0);
    let max_dist = tf.max(tf.norm(d, 'euclidean', 0));

    let pose_size = tf.maximum(tf.mul(torso_size, torso_size_multiplier), max_dist);
    return pose_size;
  }

  function normalize_pose_landmarks(landmarks) {
    let pose_center = get_center_point(landmarks, POINTS.LEFT_HIP, POINTS.RIGHT_HIP);
    pose_center = tf.expandDims(pose_center, 1);
    pose_center = tf.broadcastTo(pose_center, [1, 17, 2]);
    landmarks = tf.sub(landmarks, pose_center);

    let pose_size = get_pose_size(landmarks);
    landmarks = tf.div(landmarks, pose_size);
    return landmarks;
  }

  function landmarks_to_embedding(landmarks) {
    landmarks = normalize_pose_landmarks(tf.expandDims(landmarks, 0));
    let embedding = tf.reshape(landmarks, [1, 34]);
    return embedding;
  }

  const runMovenet = async () => {
    try {
      const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
      const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
      const poseClassifier = await tf.loadLayersModel('https://models.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json');
      
      // Store references for later use
      detectorRef.current = detector;
      classifierRef.current = poseClassifier;
      
      const countAudio = new Audio(count);
      countAudio.loop = true;
      
      // Clear any existing interval first
      if (interval) clearInterval(interval);
      
      interval = setInterval(() => {
        detectPose(detector, poseClassifier, countAudio);
      }, 100);
    } catch (error) {
      console.error("Error initializing models:", error);
    }
  };

  const detectPose = async (detector, poseClassifier, countAudio) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      try {
        let notDetected = 0;
        const video = webcamRef.current.video;
        const pose = await detector.estimatePoses(video);
        const ctx = canvasRef.current.getContext('2d');

        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvasRef.current.width, 0);

        const keypoints = pose[0].keypoints;

        let input = keypoints.map((keypoint) => {
          if (keypoint.score > 0.4) {
            if (!(keypoint.name === 'left_eye' || keypoint.name === 'right_eye')) {
              drawPoint(ctx, keypoint.x, keypoint.y, 8, 'rgb(255,255,255)');
              let connections = keypointConnections[keypoint.name];
              try {
                connections.forEach((connection) => {
                  let conName = connection.toUpperCase();
                  drawSegment(
                    ctx,
                    [keypoint.x, keypoint.y],
                    [keypoints[POINTS[conName]].x, keypoints[POINTS[conName]].y],
                    skeletonColor
                  );
                });
              } catch (err) {}
            }
          } else {
            notDetected += 1;
          }
          return [keypoint.x, keypoint.y];
        });

        if (notDetected > 4) {
          skeletonColor = 'rgb(255,255,255)';
          ctx.restore();
          return;
        }

        const processedInput = landmarks_to_embedding(input);
        const classification = poseClassifier.predict(processedInput);

        classification.array().then((data) => {
          const classNo = CLASS_NO[currentPose];
          const confidence = data[0][classNo] * 100; // Confidence as a percentage
          
          // Store current confidence globally
          currentConfidenceValue = confidence;
          // Update React state for display (less frequently)
          setCurrentConfidence(Math.round(confidence));

          if (confidence > 97) {
            if (!flag) {
              countAudio.play();
              setStartingTime(new Date(Date()).getTime());
              flag = true;
            }
            setCurrentTime(new Date(Date()).getTime());
            skeletonColor = 'rgb(0,255,0)';
          } else {
            flag = false;
            skeletonColor = 'rgb(255,255,255)';
            countAudio.pause();
            countAudio.currentTime = 0;
          }

          ctx.font = 'bold 24px Arial';
          ctx.fillStyle = 'rgb(255,255,255)';
          const canvasWidth = canvasRef.current.width;
          ctx.fillText(`Pose: ${currentPose}`, canvasWidth - 200, 50);
          ctx.fillText(`Confidence: ${Math.round(confidence)}%`, 50, 50);

          // Show training status if active
          if (isCurrentlyTraining) {
            ctx.fillText(`Training: ${countdown}s`, canvasWidth - 200, 90);
            ctx.fillText(`Samples: ${confidenceValues.length}`, 50, 90);
          }

          ctx.restore();
        });
      } catch (err) {
        console.log("Error in pose detection:", err);
      }
    }
  };

  // Function to sample confidence values at regular intervals during training
  const startSamplingConfidence = () => {
    // Clear previous samples
    confidenceValues = [];
    
    // Sample every 100ms
    samplingInterval = setInterval(() => {
      if (isCurrentlyTraining && currentConfidenceValue > 0) {
        confidenceValues.push(currentConfidenceValue);
        setSamples(confidenceValues.length);
        console.log("Added confidence:", currentConfidenceValue, "Total samples:", confidenceValues.length);
      }
    }, 100);
  };

  const startTraining = () => {
    console.log('Training started');
    
    // Set training flags
    setIsTraining(true);
    isCurrentlyTraining = true;
    
    // Reset values
    setCountdown(20);
    setScore(0);
    setSamples(0);
    confidenceValues = [];
    
    // Start sampling confidence values
    startSamplingConfidence();
    
    // Clear any existing training interval
    if (trainingInterval) clearInterval(trainingInterval);
    
    // Start countdown
    trainingInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          // End training
          clearInterval(trainingInterval);
          clearInterval(samplingInterval);
          setIsTraining(false);
          isCurrentlyTraining = false;
          
          // Calculate score 
          calculateScore();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const calculateScore = () => {
    console.log('Calculating final score...');
    console.log('Confidence values collected:', confidenceValues.length);
    
    if (confidenceValues.length === 0) {
      console.log('No confidence values recorded');
      setScore(0);
      return;
    }

    // Calculate average confidence
    const totalConfidence = confidenceValues.reduce((sum, score) => sum + score, 0);
    const averageConfidence = totalConfidence / confidenceValues.length;

    console.log('Total Confidence:', totalConfidence);
    console.log('Average Confidence:', averageConfidence);
    console.log('Samples:', confidenceValues.length);

    // Map average confidence to a score out of 10
    const finalScore = (averageConfidence / 100) * 10;
    setScore(finalScore.toFixed(2));
    setSamples(confidenceValues.length);
    console.log('Final Score:', finalScore.toFixed(2));
  };

  const stopPose = () => {
    setIsStartPose(false);
    cleanupAllIntervals();
    isCurrentlyTraining = false;
  };

  const startYoga = () => {
    setIsStartPose(true);
    runMovenet();
  };

  if (isStartPose) {
    return (
      <div className="yoga-container">
        <div className="performance-container">
          <div className="pose-performance">
            <h4>Pose Time: {poseTime} s</h4>
          </div>
          <div className="pose-performance">
            <h4>Best: {bestPerform} s</h4>
          </div>
        </div>
        <div>
          <Webcam
            width="640px"
            height="480px"
            id="webcam"
            ref={webcamRef}
            style={{
              position: 'absolute',
              left: 120,
              top: 100,
              padding: '0px',
            }}
          />
          <canvas
            ref={canvasRef}
            id="my-canvas"
            width="640px"
            height="480px"
            style={{
              position: 'absolute',
              left: 120,
              top: 100,
              zIndex: 1,
            }}
          ></canvas>
          <div>
            <img src={poseImages[currentPose]} className="pose-img" />
          </div>
        </div>
        <button onClick={stopPose} className="secondary-btn">
          Stop Pose
        </button>
        {isTraining && (
          <div className="countdown">
            <div>Time Left: {countdown}s</div>
            <div>Current Confidence: {currentConfidence}%</div>
            <div>Samples Collected: {samples}</div>
          </div>
        )}
        {!isTraining && countdown === 0 && (
          <div className="score">
            <div>Your Score: {score}/10</div>
            <div>Based on {samples} samples</div>
          </div>
        )}
        <button 
          onClick={startTraining} 
          className="secondary-btn"
          disabled={isTraining}
        >
          {isTraining ? "Training..." : "Train Me"}
        </button>
      </div>
    );
  }

  return (
    <div className="yoga-container">
      <DropDown
        poseList={poseList}
        currentPose={currentPose}
        setCurrentPose={setCurrentPose}
      />
      <Instructions currentPose={currentPose} />
      <button onClick={startYoga} className="secondary-btn">
        Start Pose
      </button>
    </div>
  );
}

export default Yoga;