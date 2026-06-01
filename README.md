# AI-Powered Foundation Shade Recommendation System

## Overview

An AI-powered foundation shade recommendation system designed to address the skin tone inclusivity gap in the Indian cosmetics industry. The system uses Computer Vision, Machine Learning, and Color Science techniques to analyze a user's skin tone and recommend the most suitable foundation shades across multiple cosmetic brands.

By leveraging facial landmark detection, dominant color extraction, and perceptual color matching, the application provides accurate and personalized cosmetic recommendations in real time.

---

## Problem Statement

Many beauty recommendation systems struggle to accurately match the diverse range of Indian skin tones, often resulting in poor product recommendations and limited inclusivity. This project aims to create a data-driven solution that delivers accurate foundation shade recommendations across multiple cosmetic brands.

---

## Features

- Real-time skin tone detection using webcam input
- Facial landmark-based skin region extraction
- Dominant skin color extraction using K-Means Clustering
- Foundation shade matching across multiple brands
- LAB color space-based similarity matching
- Personalized shade recommendations
- Interactive Flask web application
- Cross-brand foundation comparison

---

## Tech Stack

### Programming Language
- Python

### Computer Vision
- OpenCV
- Dlib

### Machine Learning
- Scikit-Learn (K-Means Clustering)

### Data Processing
- NumPy
- Pandas

### Color Science
- Scikit-Image
- SciPy

### Web Framework
- Flask

### Data Storage
- JSON

### Optimization
- Hungarian Algorithm (Linear Sum Assignment)

---

## Methodology

### 1. Data Collection
- Foundation shade information collected from cosmetic brands such as MAC, Maybelline, and L'Oréal.
- User skin tone captured through a webcam.

### 2. Skin Tone Detection
- Facial landmarks detected using Dlib's 68-point landmark predictor.
- Skin regions extracted from the face.
- K-Means clustering applied to identify the top dominant skin tones.

### 3. Product Palette Extraction
- Foundation shade data processed into RGB color palettes.
- Dominant colors extracted and stored in a structured database.

### 4. Shade Matching
- User and product palettes converted from RGB to LAB color space.
- Color distances calculated for improved perceptual accuracy.
- Hungarian Algorithm used to determine the optimal shade match.

### 5. Recommendation Generation
- Best matching foundation shades returned across supported cosmetic brands.

---

## System Workflow

```text
Webcam Input
      ↓
Face Detection
      ↓
Facial Landmark Detection
      ↓
Skin Region Extraction
      ↓
K-Means Clustering
      ↓
Dominant Skin Tone Palette
      ↓
LAB Color Space Conversion
      ↓
Foundation Palette Matching
      ↓
Personalized Shade Recommendation
```

---

## Results

- Successfully extracts dominant skin tones from facial images.
- Generates personalized foundation recommendations.
- LAB color matching produces more accurate recommendations than traditional RGB-based matching.
- Supports recommendations across multiple cosmetic brands.

---

## Project Structure

```text
Foundation-Recommendation-System/
│
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
│
├── backend/
│   ├── face_color_palette.py
│   ├── model.py
│   ├── processed_brand_palettes.json
│   ├── saved_palette.json
│   └── shape_predictor_68_face_landmarks.dat
│
└── datasets/
```

---

## Future Improvements

- Deep Learning-based skin tone estimation
- Lighting normalization and color correction
- Support for additional cosmetic brands
- Mobile application deployment
- Personalized makeup and skincare recommendations

---

## Impact

This project promotes inclusivity and personalization in the cosmetics industry by helping users discover foundation shades that accurately match their skin tone. The system demonstrates how AI and Computer Vision can be applied to solve real-world consumer challenges while encouraging diversity in beauty technology.

---

## Contributors

- Shriya Kurup
- Kavya Singh
- Khushie Agrawal
