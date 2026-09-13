# 🧠 DeepNeuro — AI Assistant Application for Brain Disease Diagnosis

**DeepNeuro** is an AI-assisted desktop application that supports doctors and radiologists in diagnosing brain diseases — **glioma tumors**, **ischemic stroke**, and (planned) **hemorrhagic stroke** — by combining secure patient/case management with deep-learning-based medical image segmentation.

This project was developed as an **End-of-Studies Internship Project** for the **Bachelor's Degree in Software Engineering and Information Systems**, Faculty of Sciences of Monastir, University of Monastir, conducted at **SmartLab, Faculty of Medicine of Monastir**.

<p align="center">
  <img width="600" src="assets/deepneuro_logo.png" alt="DeepNeuro Logo — replace with Figure 1.3" />
  <br>
  <em>Figure: DeepNeuro Platform Logo</em>
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Data Preprocessing & Augmentation](#-data-preprocessing--augmentation)
- [Model Architecture — Hybrid 3D MedNeXt U-Net](#-model-architecture--hybrid-3d-mednext-u-net)
- [Training Strategy](#-training-strategy)
- [Experimental Results](#-experimental-results)
- [Application Modules & Screenshots](#-application-modules--screenshots)
- [System Design Diagrams](#-system-design-diagrams)
- [Development Approach](#-development-approach)
- [Limitations & Future Work](#-limitations--future-work)
- [Project Team & Supervision](#-project-team--supervision)
- [References](#-references)
- [License](#-license)

---

## 🩺 Overview

Manual analysis of MRI/CT brain scans is time-consuming and subject to inter-observer variability. **DeepNeuro** addresses this by integrating:

- Secure **patient and case management** for doctors and radiologists.
- **AI-based segmentation models** built on a Hybrid 3D MedNeXt U-Net architecture.
- **2D and 3D medical image visualization** tools for diagnostic interpretation.
- A modular, three-tier architecture designed for future extension to additional pathologies.

The platform was trained and evaluated on the **BraTS 2024** (glioma) and **ISLES 2022** (ischemic stroke) datasets, achieving promising segmentation accuracy and stable training performance.

**Keywords:** Artificial Intelligence, Deep Learning, Brain Tumor Segmentation, Stroke Segmentation, Medical Imaging, MRI, U-Net, MedNeXt, BraTS 2024, ISLES 2022.

---

## 🏗️ System Architecture

DeepNeuro follows a **three-tier architecture** (Presentation, Logic, Data) to improve scalability, modularity, and maintainability.

<p align="center">
  <img width="650" src="assets/three_tier_architecture.png" alt="Three-tier architecture — replace with Figure 1.4" />
  <br>
  <em>Figure: Three-Tier Architecture</em>
</p>

| Layer | Main Responsibilities |
|---|---|
| **Presentation Layer** (Client — PySide6) | Upload MRI/CT images · Visualize segmentation results · Access patient history · Exchange diagnostic information |
| **Application Layer** (Server — Flask) | Medical image processing · AI-based segmentation · Authentication & role management · Communication between users |
| **Data Layer** (Database — MySQL) | Patient records · Medical images & segmentation outputs · User accounts & communication logs |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| **Google Colab** | Dataset preprocessing, model training, and evaluation with GPU acceleration |
| **TensorFlow** | Designing, training, and deploying deep learning segmentation models |
| **PySide6** | Graphical user interface and visualization modules |
| **Flask** | RESTful APIs and backend services |
| **MySQL** | Data persistence for users, patients, and diagnostic requests |
| **Postman** | API testing and debugging |
| **Visual Studio Code** | Development, debugging, and project management |
| **Git** | Version control |

---

## ✨ Key Features

| ID | Feature | Description |
|---|---|---|
| FR1 | User Registration | Doctors and radiologists create accounts with personal/professional information |
| FR2 | User Authentication | Secure login/logout with role-based access |
| FR3 | Patient Record Management | Doctors manage patient records and submit diagnostic cases |
| FR4 | Image Upload | Radiologists/doctors upload MRI and CT medical images |
| FR5 | Automatic Segmentation | AI-based segmentation of glioma tumors, ischemic stroke lesions, and hemorrhagic stroke regions |
| FR6 | Result Visualization & Validation | Radiologists validate diagnostic outputs before sharing with doctors |
| FR7 | Professional Communication | Communication between doctors and radiologists within the platform |

**Non-functional highlights:** data confidentiality & secure access control · acceptable inference time for clinical use · modular architecture for future AI model integration · intuitive UI for medical professionals.

---

## 🗂️ Datasets

### 1️⃣ BraTS 2024 — Glioma Tumor Segmentation

~1500 patient samples (≈300 validation cases). Each sample includes four co-registered MRI modalities of shape **182×218×182**:

- **T1-weighted (T1n):** anatomical brain structure.
- **T1-contrast enhanced (T1c):** highlights enhancing tumor regions.
- **T2-weighted (T2w):** edema and tumor boundaries.
- **FLAIR (T2f):** suppresses fluid signal, enhances lesion visibility.

<p align="center">
  <img width="700" src="assets/brats_modalities.png" alt="BraTS 2024 MRI modalities — replace with Figure 5.3 / README Fig T1n-T1c-T2f-T2w" />
  <br>
  <em>Figure: BraTS 2024 MRI Modalities for the Same Patient</em>
</p>

**Segmentation classes:**

| Class | Label |
|---|---|
| 0 | Background |
| 1 | Necrotic / Non-enhancing tumor core |
| 2 | Peritumoral edema |
| 3 | Enhancing tumor |
| 4 | Resection cavity |

<p align="center">
  <img width="700" src="assets/brats_segmentation_mask.png" alt="BraTS segmentation mask overlay — replace with Figure 5.4" />
  <br>
  <em>Figure: BraTS 2024 Segmentation Mask Example (T1n MRI with Tumor Overlay)</em>
</p>

### 2️⃣ ISLES 2022 — Ischemic Stroke Lesion Segmentation

~300 patient samples (≈50 validation cases) from multiple clinical centers.

- **DWI** (112×112×73): highlights restricted water diffusion (acute stroke).
- **ADC** (112×112×73): quantifies diffusion intensity.
- **FLAIR** (281×352×352): additional structural information (requires resizing).

<p align="center">
  <img width="700" src="assets/isles_modalities.png" alt="ISLES 2022 DWI/ADC modalities — replace with Figure 5.5" />
  <br>
  <em>Figure: ISLES 2022 DWI and ADC MRI Modalities</em>
</p>

**Segmentation classes:**

| Class | Label |
|---|---|
| 0 | Background |
| 1 | Ischemic stroke lesion |

<p align="center">
  <img width="700" src="assets/isles_segmentation_mask.png" alt="ISLES stroke mask overlay — replace with Figure 5.6" />
  <br>
  <em>Figure: ISLES 2022 Stroke Segmentation Mask Example (ADC with Overlay)</em>
</p>

---

## 🔄 Data Preprocessing & Augmentation

**Preprocessing pipeline:**

1. **Loading** `.nii.gz` volumes via `nibabel`.
2. **Z-score normalization** per modality.
3. **Resizing** MRI volumes to a fixed shape (linear interpolation, order=1).
4. **Resizing masks** with nearest-neighbor interpolation (order=0) to preserve discrete labels.
5. **Multi-modal stacking** into a single 4D volume (e.g., 128×128×128×4 for BraTS).

<p align="center">
  <img width="750" src="assets/preprocessing_workflow.png" alt="Preprocessing & augmentation workflow — replace with the BraTS24 workflow diagram" />
  <br>
  <em>Figure: Data Processing and Augmentation Workflow</em>
</p>

**Augmentation techniques** (applied jointly to volumes and masks to preserve spatial consistency):

- Random 3D flipping
- Random 90° rotations
- Intensity scaling
- Gaussian noise injection
- Gamma correction

<p align="center">
  <img width="750" src="assets/augmentation_examples.png" alt="Data augmentation examples — replace with Figure 5.7" />
  <br>
  <em>Figure: Examples of Data Augmentation Applied to Medical Image Slices</em>
</p>

---

## 🧠 Model Architecture — Hybrid 3D MedNeXt U-Net

DeepNeuro adopts a **3D Hybrid U-Net** combining a classic encoder–bottleneck–decoder structure with **MedNeXt-style convolutional blocks** and **adaptive normalization**.

<p align="center">
  <img width="800" src="assets/hybrid_3d_mednext_unet.png" alt="Hybrid 3D MedNeXt U-Net architecture — replace with Figure 5.8" />
  <br>
  <em>Figure: Hybrid 3D MedNeXt U-Net Architecture</em>
</p>

**Key design choices:**

- **Encoder–Decoder** with skip connections (4 downsampling / 4 upsampling stages).
- **MedNeXt-inspired blocks:** depthwise/pointwise convolutions, residual connections, Squeeze-and-Excitation (SE) attention.
- **Adaptive Group Normalization:** stable under the small batch sizes required for 3D volumetric training.
- **3D convolutions** throughout for full volumetric feature extraction.
- **Dropout regularization** for generalization.
- **Softmax output layer** for voxel-wise multi-class segmentation.

---

## 🏋️ Training Strategy

| Aspect | Details |
|---|---|
| **Optimizers** | Adam and SGD (with momentum) |
| **Loss function** | Hybrid: `α × Sparse Categorical Cross-Entropy + β × Dice Loss` |
| **Metric** | Multiclass Dice Similarity Coefficient (DSC) |
| **Epochs** | Up to 50, with early stopping |
| **Callbacks** | `EarlyStopping` (patience 5) · `ModelCheckpoint` (best val loss) · `CSVLogger` · `ReduceLROnPlateau` (factor 0.5, patience 3) |

Dice coefficient:

```
Dice = (2 × |P ∩ G| + ε) / (|P| + |G| + ε)
```

---

## 📊 Experimental Results

### BraTS 2024 (Glioma)

| Input Size | Batch Size | (α, β) | Val Loss | Val DSC |
|---|---|---|---|---|
| 96×96×96×4 | 4 | (1, 1) | 0.421 | 0.681 |
| 128×128×128×4 | 3 | (2, 3) | 0.337 | 0.742 |
| **160×192×160×4** | **2** | **(4, 5)** | **0.284** | **0.791** ✅ |

<p align="center">
  <img width="800" src="assets/brats_training_curves.png" alt="BraTS training/validation DSC & loss curves — replace with Figure 5.9" />
  <br>
  <em>Figure: Training and Validation DSC and Loss Curves — BraTS 2024</em>
</p>

### ISLES 2022 (Ischemic Stroke)

| Modalities | Input Size | Batch Size | (α, β) | Val Loss | Val DSC |
|---|---|---|---|---|---|
| ADC + DWI + FLAIR | 90×90×60×3 | 4 | (1, 1) | 0.463 | 0.641 |
| ADC + DWI | 112×112×72×2 | 5 | (2, 3) | 0.351 | 0.784 |
| **ADC + DWI** | **112×112×72×2** | **5** | **(4, 5)** | **0.287** | **0.851** ✅ |

<p align="center">
  <img width="800" src="assets/isles_training_curves.png" alt="ISLES training/validation DSC & loss curves — replace with Figure 5.10" />
  <br>
  <em>Figure: Training and Validation DSC and Loss Curves — ISLES 2022</em>
</p>

---

## 🖥️ Application Modules & Screenshots

### 🔐 Authentication System

Secure sign-up, login, email verification (temporary codes), password hashing, and role-based access control (Doctor / Radiologist).

<table align="center">
  <tr>
    <td align="center"><b>Registration</b></td>
    <td align="center"><b>Duplicate Email Error</b></td>
  </tr>
  <tr>
    <td><img width="380" src="assets/registration_interface.png" alt="User Registration Interface — Figure 3.4" /></td>
    <td><img width="380" src="assets/duplicate_email_error.png" alt="Duplicate Email Error — Figure 3.5" /></td>
  </tr>
  <tr>
    <td align="center"><b>Login</b></td>
    <td align="center"><b>Email Verification</b></td>
  </tr>
  <tr>
    <td><img width="380" src="assets/login_interface.png" alt="User Login Interface — Figure 3.6" /></td>
    <td><img width="380" src="assets/email_verification_interface.png" alt="Email Verification Interface — Figure 3.7" /></td>
  </tr>
</table>

### 🗃️ Patient & Case Management

Doctor dashboard, patient record CRUD, case submission to radiologists, and medical history tracking.

<p align="center">
  <img width="700" src="assets/doctor_dashboard.png" alt="Doctor Dashboard Interface — Figure 4.4" />
  <br>
  <em>Figure: Doctor Dashboard Interface</em>
</p>

**2D Medical Image Visualization** — four independent panels for multi-modality comparison, drag-and-drop file loading, slice navigation slider, multiple colormaps (grayscale, jet, hot, etc.), and side-by-side comparison of scans across dates.

<p align="center">
  <img width="750" src="assets/2d_visualization_interface.png" alt="2D Medical Image Visualization Interface — Figure 4.5" />
  <br>
  <em>Figure: 2D Medical Image Visualization Interface</em>
</p>

**3D Segmentation Visualization** — interactive 3D brain models, adjustable layer opacity, color-coded tumor subregions (necrotic core: brown, edema: green, enhancing tumor: red, resection cavity: blue), and a quantitative statistics panel (brain/tumor voxels, tumor volume).

<p align="center">
  <img width="750" src="assets/3d_visualization_interface.png" alt="3D Segmentation Mask Visualization Interface — Figure 4.6" />
  <br>
  <em>Figure: 3D Segmentation Mask Visualization Interface</em>
</p>

### 🩻 AI Diagnostic Workflow

Case information review and MRI upload → preprocessing → AI inference → segmentation mask generation and delivery back to the doctor.

<table align="center">
  <tr>
    <td align="center"><b>Case Information</b></td>
    <td align="center"><b>Segmentation Workflow</b></td>
  </tr>
  <tr>
    <td><img width="380" src="assets/case_information_interface.png" alt="Case Information Interface — Figure 5.11a" /></td>
    <td><img width="380" src="assets/segmentation_workflow_interface.png" alt="Segmentation Workflow Interface — Figure 5.11b" /></td>
  </tr>
</table>

---

## 📐 System Design Diagrams

<p align="center">
  <img width="800" src="assets/global_use_case_diagram.png" alt="Global Use Case Diagram — Figure 2.1" />
  <br>
  <em>Global Use Case Diagram of the DeepNeuro System</em>
</p>

<p align="center">
  <img width="800" src="assets/global_class_diagram.png" alt="Global Class Diagram — Figure 2.2" />
  <br>
  <em>DeepNeuro Global Class Diagram (User, Doctor, Radiologist, Patient, DiagnosisRequest, FileUpload)</em>
</p>

<details>
<summary>📄 Additional sequence diagrams (click to expand placeholders)</summary>

<p align="center">
  <img width="700" src="assets/signup_sequence_diagram.png" alt="Sign Up Sequence Diagram — Figure 3.2" /><br>
  <em>Sign Up Sequence Diagram</em>
</p>
<p align="center">
  <img width="700" src="assets/login_sequence_diagram.png" alt="Login Sequence Diagram — Figure 3.3" /><br>
  <em>Login Sequence Diagram</em>
</p>
<p align="center">
  <img width="700" src="assets/add_patient_sequence_diagram.png" alt="Add Patient Sequence Diagram — Figure 4.2" /><br>
  <em>Add Patient Sequence Diagram</em>
</p>
<p align="center">
  <img width="700" src="assets/send_case_sequence_diagram.png" alt="Send Case Sequence Diagram — Figure 4.3" /><br>
  <em>Send Case Sequence Diagram</em>
</p>
<p align="center">
  <img width="700" src="assets/segmentation_workflow_sequence_diagram.png" alt="Segmentation Workflow Sequence Diagram — Figure 5.2" /><br>
  <em>Segmentation Workflow Sequence Diagram</em>
</p>

</details>

---

## 🛠️ Development Approach

The project followed an **Agile-inspired methodology** organized into three sprints:

<p align="center">
  <img width="700" src="assets/sprint_planning_strategy.png" alt="Sprint Planning Strategy — Figure 2.3" />
</p>

| Sprint | Focus |
|---|---|
| **Sprint 1** | Authentication System Development (registration, login, email verification, RBAC) |
| **Sprint 2** | Patient & Case Management Module (patient records, case sharing, 2D/3D visualization) |
| **Sprint 3** | Segmentation Models Integration (data pipeline, model training, AI inference integration) |

Git and Postman supported version control and frontend–backend communication testing.

---

## 🚧 Limitations & Future Work

- **Hemorrhagic stroke segmentation** was not implemented due to computational/time constraints — the architecture is designed to support it in the future.
- Planned improvements: inference-time optimization, support for additional brain diseases, **Explainable AI (XAI)** integration for prediction transparency, cloud-based deployment, real-time collaborative workflows, and enhanced visualization/reporting.

---

## 👨‍💻 Project Team & Supervision

| Role | Name |
|---|---|
| **Prepared by** | Mohamed Aziz Krifa |
| **Academic Supervisor** | Hela Haj Mohamed |
| **Professional Supervisor** | Racha Zaibi |
| **Jury President** | Hend Basly |
| **Reviewer** | Adnen Mahmoud |
| **Host Institution** | SmartLab, Faculty of Medicine of Monastir |

---

## 📚 References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015.
- Çiçek et al., *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*, 2016.
- Oktay et al., *Attention U-Net*, 2018.
- Roy et al., *MedNeXt: Transformer-Driven Scaling of ConvNets for Medical Image Segmentation*, 2023.
- Wu & He, *Group Normalization*, 2018.
- He et al., *Deep Residual Learning for Image Recognition*, 2016.
- Hu et al., *Squeeze-and-Excitation Networks*, 2018.
- Milletari et al., *V-Net: Fully Convolutional Networks for Volumetric Medical Image Segmentation*, 2016.
- [BraTS 2024 Challenge](https://www.med.upenn.edu/cbica/brats2024/)
- [ISLES Challenge](https://www.isles-challenge.org/)

---

## 📄 License

This project was developed for academic purposes as part of an end-of-studies internship. Add your preferred license (e.g., MIT) here.

---

<p align="center">Made with 🧠 by <b>Mohamed Aziz Krifa</b> — SmartLab, Faculty of Medicine of Monastir</p>
