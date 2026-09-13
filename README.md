# 🧠 DeepNeuro — AI Assistant Application for Brain Disease Diagnosis

**DeepNeuro** is an AI-assisted desktop application that supports doctors and radiologists in diagnosing brain diseases — **glioma tumors**, **ischemic stroke**, and (planned) **hemorrhagic stroke** — by combining secure patient/case management with deep-learning-based medical image segmentation.

This project was developed as an **End-of-Studies Internship Project** for the **Bachelor's Degree in Software Engineering and Information Systems**, Faculty of Sciences of Monastir, University of Monastir, conducted at **SmartLab, Faculty of Medicine of Monastir**.

<p align="center">
  <img width="300" src="https://github.com/user-attachments/assets/c870faad-5488-402a-84cf-2dab850a4e50" />
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
  <img width="900" src="https://github.com/user-attachments/assets/86020e58-8e8d-4965-8759-119e68209ad9" />
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
<table align="center" >
  <tr>
    <td colspan="2" align="center">
      <h3>MRI Tests on the Same Patient X</h3>
    </td>
  </tr>

  <tr>
    <td align="center"><b>T1n</b></td>
    <td align="center"><b>T1c</b></td>
  </tr>
  <tr>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/a442156b-c73e-4b4e-a279-7257d8ac633d" /></td>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/bc7c7471-084f-4347-b3c4-2c27284a7333" /></td>
  </tr>
  <tr>
    <td align="center"><b>T2f</b></td>
    <td align="center"><b>T2w</b></td>
  </tr>
  <tr>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/b27657c9-327d-49ae-88c9-06944b982e9e" /></td>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/b02966ed-05f7-45ba-97fc-fe25992c8a32" /></td>
  </tr>
</table>

**Segmentation classes:**

| Class | Label |
|---|---|
| 0 | Background |
| 1 | Necrotic / Non-enhancing tumor core |
| 2 | Peritumoral edema |
| 3 | Enhancing tumor |
| 4 | Resection cavity |

<p align="center">
  <img width="900" src="https://github.com/user-attachments/assets/6d4779e8-eb8e-496a-9d82-93009fdc764b" />
</p>

### 2️⃣ ISLES 2022 — Ischemic Stroke Lesion Segmentation

~300 patient samples (≈50 validation cases) from multiple clinical centers.

- **DWI** (112×112×73): highlights restricted water diffusion (acute stroke).
- **ADC** (112×112×73): quantifies diffusion intensity.
- **FLAIR** (281×352×352): additional structural information (requires resizing).

<p align="center">
  <img width="900" src="https://github.com/user-attachments/assets/0c6352e3-5ef4-4e55-9ef3-c735ada31d0c"/>
</p>

**Segmentation classes:**

| Class | Label |
|---|---|
| 0 | Background |
| 1 | Ischemic stroke lesion |

<p align="center">
  <img width="900" src="https://github.com/user-attachments/assets/cd762481-9344-426d-a3e7-82ecfadebf0e" />
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
  <img width="900" src="https://github.com/user-attachments/assets/430772cf-7b10-4949-a766-48feb7065b5a" />
</p>

**Augmentation techniques** (applied jointly to volumes and masks to preserve spatial consistency):

- Random 3D flipping
- Random 90° rotations
- Intensity scaling
- Gaussian noise injection
- Gamma correction

<table align="center">
  <tr>
    <td align="center"><b>Image 1</b></td>
    <td align="center"><b>Image 2</b></td>
  </tr>
  <tr>
    <td>
      <img width="600" height="400"
           src="https://github.com/user-attachments/assets/6e214cd2-0317-4955-b295-3f1be7010c05" />
    </td>
    <td>
      <img width="600" height="400"
           src="https://github.com/user-attachments/assets/68852274-1459-40a4-98cd-f4dfc479c0d6" />
    </td>
  </tr>

  <tr>
    <td align="center"><b>Image 3</b></td>
    <td align="center"><b>Image 4</b></td>
  </tr>
  <tr>
    <td>
      <img width="600" height="400"
           src="https://github.com/user-attachments/assets/1cdebeda-da35-4a5b-a887-b7a78527b843" />
    </td>
    <td>
      <img width="600" height="400"
           src="https://github.com/user-attachments/assets/dd95fa6d-cd63-4f05-8397-2f061af9b18b" />
    </td>
  </tr>
</table>

---

## 🧠 Model Architecture — Hybrid 3D MedNeXt U-Net

DeepNeuro adopts a **3D Hybrid U-Net** combining a classic encoder–bottleneck–decoder structure with **MedNeXt-style convolutional blocks** and **adaptive normalization**.

<p align="center">
  <img width="900" alt="Hybrid 3D MedNeXt U-Net Arch" src="https://github.com/user-attachments/assets/dc848d61-42a6-4863-bd0c-5fdb24bd1785" />
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
  <img width="800" src="https://github.com/user-attachments/assets/77c1dc18-0a81-427c-8a83-77cea8fbc175"/>
</p>

### ISLES 2022 (Ischemic Stroke)

| Modalities | Input Size | Batch Size | (α, β) | Val Loss | Val DSC |
|---|---|---|---|---|---|
| ADC + DWI + FLAIR | 90×90×60×3 | 4 | (1, 1) | 0.463 | 0.641 |
| ADC + DWI | 112×112×72×2 | 5 | (2, 3) | 0.351 | 0.784 |
| **ADC + DWI** | **112×112×72×2** | **5** | **(4, 5)** | **0.287** | **0.851** ✅ |

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/b8327197-ef80-4c23-afd1-e331b594a627" />
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
    <td><img width="380" src="https://github.com/user-attachments/assets/ac841173-1651-49ca-8698-d99878438d4e" /></td>
    <td><img width="380" src="https://github.com/user-attachments/assets/e4520830-5a3b-44c0-bc0f-2bab89e7f37b" /></td>
  </tr>
  <tr>
    <td align="center"><b>Login</b></td>
    <td align="center"><b>Email Verification</b></td>
  </tr>
  <tr>
    <td><img width="380" src="https://github.com/user-attachments/assets/f6d465f6-043c-4c65-bca6-cfd9b4838753"/></td>
    <td><img width="380" src="assets/email_verification_interface.png" alt="Email Verification Interface — Figure 3.7" /></td>
    <img width="868" height="914" alt="image" src="https://github.com/user-attachments/assets/dc2fe6e5-6691-43d3-9872-ffb709439298" />

  </tr>
</table>

### 🗃️ Patient & Case Management

Doctor dashboard, patient record CRUD, case submission to radiologists, and medical history tracking.

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/fdb3584d-1055-4c89-8c28-3fe652f161f0" />
</p>

**2D Medical Image Visualization** — four independent panels for multi-modality comparison, drag-and-drop file loading, slice navigation slider, multiple colormaps (grayscale, jet, hot, etc.), and side-by-side comparison of scans across dates.

<p align="center">
  <img width="750" src="https://github.com/user-attachments/assets/b5a92603-dca3-4527-874a-e1758b2e59d5" />
</p>

**3D Segmentation Visualization** — interactive 3D brain models, adjustable layer opacity, color-coded tumor subregions (necrotic core: brown, edema: green, enhancing tumor: red, resection cavity: blue), and a quantitative statistics panel (brain/tumor voxels, tumor volume).

<p align="center">
  <img width="750" src="https://github.com/user-attachments/assets/71164b82-d682-4b4b-b700-aa760dd102ca" />
</p>

### 🩻 AI Diagnostic Workflow

Case information review and MRI upload → preprocessing → AI inference → segmentation mask generation and delivery back to the doctor.

<table align="center">
  <tr>
    <td align="center"><b>Case Information</b></td>
    <td align="center"><b>Segmentation Workflow</b></td>
  </tr>
  <tr>
    <td><img width="380" src="https://github.com/user-attachments/assets/e517b1d2-ea30-40fe-bfca-74cbb573209c" /></td>
    <td><img width="380" src="https://github.com/user-attachments/assets/5880a8f7-aa0f-4738-966c-caadb1041a50" /></td>
  </tr>
</table>

---

## 🛠️ Development Approach

The project followed an **Agile-inspired methodology** organized into three sprints:

<p align="center">
  <img width="900" src="https://github.com/user-attachments/assets/4f39174d-08a0-4b2d-8984-64ffad0176b8" />
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
