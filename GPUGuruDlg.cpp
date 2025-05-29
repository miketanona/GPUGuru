// GPUGuruDlg.cpp : implementation file

#include "pch.h"
#include "framework.h"
#include "GPUGuru.h"
#include "GPUGuruDlg.h"
#include "afxdialogex.h"
#include "GpuFilters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

const CString kAppTitle = _T("GPU Guru");

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX) {}
void CAboutDlg::DoDataExchange(CDataExchange* pDX) { CDialogEx::DoDataExchange(pDX); }
BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

BEGIN_MESSAGE_MAP(CGPUGuruDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_CTLCOLOR()
	ON_WM_TIMER()
	ON_WM_QUERYDRAGICON()
	ON_WM_ERASEBKGND()
	ON_BN_CLICKED(IDC_CB_GPU, &CGPUGuruDlg::OnBnClickedCbGpu)
	ON_CBN_SELCHANGE(IDC_CB_FILTER, &CGPUGuruDlg::OnCbnSelchangeCbFilter)
	ON_BN_CLICKED(IDOK, &CGPUGuruDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &CGPUGuruDlg::OnBnClickedCancel)
	ON_BN_CLICKED(IDC_STOP, &CGPUGuruDlg::OnBnClickedStop)
END_MESSAGE_MAP()

CGPUGuruDlg::CGPUGuruDlg(CWnd* pParent)
	: CDialogEx(IDD_GPUGURU_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDI_ICON_GURU);
}


void CGPUGuruDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_CBIndex(pDX, IDC_CB_FILTER, (int&)m_filterType);
	DDX_Radio(pDX, IDC_RADIO_CV, m_processingMode);
}

BOOL CGPUGuruDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	GetDlgItem(IDC_PICTURE)->ShowWindow(SW_HIDE);

	GetDlgItem(IDC_RADIO_HYBRID)->EnableWindow(FALSE);

	m_imageDisplay.SubclassDlgItem(IDC_PICTURE, this);

	//int deviceCount = 0;
	//try {
	//	deviceCount = cv::cuda::getCudaEnabledDeviceCount();
	//	TRACE(_T("CUDA devices found: %d\n"), deviceCount);
	//	return TRUE;
	//}
	//catch (const cv::Exception& e) {
	//	TRACE(_T("CUDA error: %s\n"), CString(e.what()));
	//	return TRUE;
	//}

	SetIcon(m_hIcon, TRUE);
	SetIcon(m_hIcon, FALSE);
	SetWindowText(kAppTitle);

	// Add About menu
	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set default filter
	if (CComboBox* pCombo = (CComboBox*)GetDlgItem(IDC_CB_FILTER))
		pCombo->SetCurSel(0);

	return TRUE;
}

BOOL CGPUGuruDlg::OnEraseBkgnd(CDC* pDC)
{
	CRect rect;
	GetClientRect(&rect);
	CBrush brush(RGB(225, 235, 245));
	pDC->FillRect(&rect, &brush);
	return TRUE;
}

void CGPUGuruDlg::ResizeDialogToFitCamera()
{
	cv::Mat tempFrame;
	cap >> tempFrame;
	if (tempFrame.empty()) return;

	int width = tempFrame.cols;
	int height = tempFrame.rows;

	CRect pictureRect;
	m_imageDisplay.GetWindowRect(&pictureRect);
	ScreenToClient(&pictureRect);
	m_imageDisplay.MoveWindow(pictureRect.left, pictureRect.top, width, height);

	int marginX = pictureRect.left;
	int marginY = pictureRect.top;
	SetWindowPos(nullptr, 0, 0,
		width + 2 * marginX,
		height + 2 * marginY + GetSystemMetrics(SM_CYCAPTION),
		SWP_NOMOVE | SWP_NOZORDER);
}


void CGPUGuruDlg::OnTimer(UINT_PTR nIDEvent)
{
	UpdateData(TRUE);
	cv::Mat result;

	if (cap.read(frame))
	{
		auto start = std::chrono::high_resolution_clock::now();
		result = ProcessFrame(frame);
		auto end = std::chrono::high_resolution_clock::now();

		double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
		totalTime += elapsed;
		frameCount++;

		if (frameCount >= maxFrames)
		{
			double avg = totalTime / frameCount;
			UpdateTitleWithMetrics(avg);
			totalTime = 0.0;
			frameCount = 0;
		}

		cv::Mat rgb;
		cv::cvtColor((m_filterType == FILTER_NONE ? frame : result), rgb, cv::COLOR_BGR2RGB);

		CImage image;
		image.Create(rgb.cols, rgb.rows, 24);
		for (int y = 0; y < rgb.rows; y++) {
			BYTE* dst = (BYTE*)image.GetPixelAddress(0, y);
			BYTE* src = rgb.ptr(y);
			memcpy(dst, src, rgb.cols * 3);
		}
		CClientDC dc(&m_imageDisplay);
		image.Draw(dc, 0, 0);
	}

	CDialogEx::OnTimer(nIDEvent);
}

cv::Mat CGPUGuruDlg::ApplyCpuBlur(const cv::Mat& input)
{
	cv::Mat output;
	cv::GaussianBlur(input, output, cv::Size(9, 9), 2.0);
	return output;
}


cv::Mat CGPUGuruDlg::ApplyCpuInvert(const cv::Mat& input) {

	cv::Mat output = 255 - input;

	return output;
}



cv::Mat CGPUGuruDlg::ApplyCpuFilterEdge(const cv::Mat& input)
{
	cv::Mat gray, gradX, gradY, absGradX, absGradY, output;
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
	cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);
	cv::convertScaleAbs(gradX, absGradX);
	cv::convertScaleAbs(gradY, absGradY);
	cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, output);
	cv::cvtColor(output, output, cv::COLOR_GRAY2RGB);
	return output;
}

cv::Mat CGPUGuruDlg::ProcessFrame(const cv::Mat& input)
{
	if (m_filterType == FILTER_NONE) return input.clone();

	cv::Mat gray;

	if (m_filterType == FILTER_INVERT)
	{
		cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	}

	//TRACE("ProcessingMode = %d\n", m_processingMode);

	if (m_processingMode == MODE_GPU_ONLY) {
		if (m_filterType == FILTER_EDGE) return ApplyCudaFilterEdge(input);
		if (m_filterType == FILTER_BLUR) return ApplyCudaBlur(input);
		if (m_filterType == FILTER_INVERT) return ApplyCudaInvert(gray);
	}
	else if (m_processingMode == MODE_CV_ONLY) 
	{
		if (m_filterType == FILTER_EDGE) return ApplyCpuFilterEdge(input);
		if (m_filterType == FILTER_BLUR) return ApplyCpuBlur(input);
		if (m_filterType == FILTER_INVERT) return ApplyCpuInvert(gray);
	}
	else if (m_processingMode == MODE_CV_WITH_GPU) 
	{

	}

	return input.clone(); // fallback
}

void CGPUGuruDlg::UpdateTitleWithMetrics(double avgTime)
{
	CString mode = _T("GPU");  

	if (m_processingMode == MODE_GPU_ONLY) {
		mode = _T("GPU");   
	}
	else if (m_processingMode == MODE_CV_ONLY) {
		mode = _T("CV");
	}
	else {
		mode = _T("CV + GPU");  
	}


	TRACE(_T("PROCESSING MODE: %s\n"), mode.GetString());

	CString filter;
	switch (m_filterType) {
	case FILTER_NONE:  filter = _T("None"); break;
	case FILTER_EDGE:  filter = _T("Edge"); break;
	case FILTER_BLUR:  filter = _T("Blur"); break;
	default:           filter = _T("Unknown"); break;
	}

	CString title;
	title.Format(_T("%s | Filter: %s | Mode: %s | Avg time: %.2f ms"),
		kAppTitle, filter, mode, avgTime);
	SetWindowText(title);
}

HCURSOR CGPUGuruDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CGPUGuruDlg::OnBnClickedCbGpu()
{
	totalTime = 0.0;
	frameCount = 0;
	SetWindowText(kAppTitle + _T(" | GPU mode changed — timing reset"));
}

void CGPUGuruDlg::OnCbnSelchangeCbFilter()
{
	totalTime = 0.0;
	frameCount = 0;
	SetWindowText(kAppTitle + _T(" | Filter changed — timing reset"));
}


void CGPUGuruDlg::OnBnClickedOk()
{
	UpdateData(TRUE);

	GetDlgItem(IDC_PICTURE)->ShowWindow(SW_SHOW);

	cap.open(0);
	if (!cap.isOpened()) {
		AfxMessageBox(_T("Cannot open camera"));
		return;
	}

	ResizeDialogToFitCamera();
	SetTimer(101, 30, nullptr);  // 30ms interval (about 33 FPS)

}

void CGPUGuruDlg::OnBnClickedCancel()
{
	CDialogEx::OnCancel();
}

void CGPUGuruDlg::OnBnClickedStop()
{
	KillTimer(101);      // stop frame update loop
	cap.release();       // release webcam

	// Optional: clear image from window
	m_imageDisplay.SetBitmap(nullptr);
}
