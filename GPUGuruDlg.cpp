
// GPUGuruDlg.cpp : implementation file
//


#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "framework.h"
#include "GPUGuru.h"
#include "GPUGuruDlg.h"
#include "afxdialogex.h"
#include "GpuFilters.h"



cv::Mat ApplyCudaFilter(const cv::Mat& input);

const CString kAppTitle = _T("GPU Guru ");

void TestColorConvert(cv::Mat& input) {
	cv::Mat output;
	cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
}

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

extern void RunDummyCudaKernel();

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()




CGPUGuruDlg::CGPUGuruDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_GPUGURU_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDI_ICON_GURU);
}

void CGPUGuruDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_CB_GPU, m_useGPU);
	DDX_CBIndex(pDX, IDC_CB_FILTER, (int&)m_filterType);
}

BEGIN_MESSAGE_MAP(CGPUGuruDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_TIMER()
	ON_WM_QUERYDRAGICON()
	//ON_CBN_SELCHANGE(IDC_CB_FILTER, &CGPUGuruDlg::OnCbnSelchangeCbFilter)
	ON_BN_CLICKED(IDC_CB_GPU, &CGPUGuruDlg::OnBnClickedCbGpu)
	ON_CBN_SELCHANGE(IDC_CB_FILTER, &CGPUGuruDlg::OnCbnSelchangeCbFilter)
END_MESSAGE_MAP()




// CGPUGuruDlg message handlers

BOOL CGPUGuruDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	CComboBox* pCombo = (CComboBox*)GetDlgItem(IDC_CB_FILTER);
	if (pCombo)
	{
		pCombo->SetCurSel(0);  // 
	}

	const CString kAppTitle = _T("GPU Image Processor");

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	//RunDummyCudaKernel();

	SetWindowText(kAppTitle);


	m_imageDisplay.SubclassDlgItem(IDC_PICTURE, this);

	cap.open(0); // open default camera
	if (!cap.isOpened()) {
		AfxMessageBox(_T("Cannot open camera"));
		return FALSE;
	}

	ResizeDialogToFitCamera();

	//TRACE("SetTimer\n");
	//SetTimer(1, 30, nullptr); // 30ms ~ 33 FPS
	SetTimer(101, 30, nullptr);

	return TRUE;  // return TRUE  unless you set the focus to a control
}


BOOL CGPUGuruDlg::OnEraseBkgnd(CDC* pDC)
{
	// Example: soft blue-gray science color
	CRect rect;
	GetClientRect(&rect);

	// Choose your color — subtle blue or gray
	COLORREF bgColor = RGB(225, 235, 245); // light science-gray-blue
	CBrush brush(bgColor);
	pDC->FillRect(&rect, &brush);

	return TRUE;
}


void CGPUGuruDlg::ResizeDialogToFitCamera()
{
	cv::Mat frame;
	cap >> frame;

	if (frame.empty()) return;

	int width = frame.cols;
	int height = frame.rows;

	CRect pictureRect;
	m_imageDisplay.GetWindowRect(&pictureRect);
	ScreenToClient(&pictureRect);

	m_imageDisplay.MoveWindow(pictureRect.left, pictureRect.top, width, height);

	// Resize dialog to fit around image
	int marginX = pictureRect.left;
	int marginY = pictureRect.top;

	SetWindowPos(nullptr, 0, 0,
		width + 2 * marginX,
		height + 2 * marginY + GetSystemMetrics(SM_CYCAPTION),
		SWP_NOMOVE | SWP_NOZORDER);
}




void CGPUGuruDlg::ResizeWindow()
{

	cv::Mat tempFrame;
	cap >> tempFrame;

	if (!tempFrame.empty())
	{
		int width = tempFrame.cols;
		int height = tempFrame.rows;

		// Resize the CStatic control to fit the frame
		CRect pictureRect;
		m_imageDisplay.GetWindowRect(&pictureRect);
		ScreenToClient(&pictureRect);

		m_imageDisplay.MoveWindow(pictureRect.left, pictureRect.top, width, height);

		// Resize the whole dialog to fit around the image
		CRect dialogRect;
		GetWindowRect(&dialogRect);
		ScreenToClient(&dialogRect);

		int marginX = pictureRect.left;
		int marginY = pictureRect.top;

		SetWindowPos(nullptr, 0, 0,
			width + 2 * marginX,
			height + 2 * marginY + GetSystemMetrics(SM_CYCAPTION),
			SWP_NOMOVE | SWP_NOZORDER);
	}

}




void CGPUGuruDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CGPUGuruDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}


void CGPUGuruDlg::OnTimer(UINT_PTR nIDEvent)
{
	UpdateData(TRUE); // reads checkbox state into m_useGPU



	//TRACE(_T("Filter enum value = %d USE GPU = %d\n"), static_cast<int>(m_filterType), static_cast<int>(m_useGPU));

	cv::Mat result;

	if (cap.read(frame)) {
		cv::Mat result;

		// Apply either CPU or GPU filter
		//if (m_useGPU) {
		//	result = ApplyCudaFilter(frame);  // Implemented in GpuFilters.cu
		//}
		//else {
		//	result = ApplyCpuFilter(frame);   // Uses OpenCV
		//}

		// Convert to RGB for display

		auto start = std::chrono::high_resolution_clock::now();

		if (m_useGPU)
		{
			switch (m_filterType)
			{
			case FILTER_EDGE:
				result = ApplyCudaFilter(frame);
				break;
			case FILTER_BLUR:
				result = ApplyCudaBlur(frame);
				//result = frame.clone(); // TODO: Replace with CUDA blur
				break;
			}
		}
		else
		{
			switch (m_filterType)
			{
			case FILTER_EDGE:
				result = ApplyCpuFilter(frame);
				break;
			case FILTER_BLUR:
				result = ApplyCpuBlur(frame);
				break;
			}
		}

		auto end = std::chrono::high_resolution_clock::now();

		double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

		// Update running average
		totalTime += elapsed;
		frameCount++;

		if (frameCount >= maxFrames)
		{
			double avg = totalTime / frameCount;

			CString title;
			title.Format(_T("Using %s - Avg time: %.2f ms (over %d frames)"),
				m_useGPU ? _T("GPU") : _T("CPU"),
				avg, frameCount);
			SetWindowText(title);

			// Reset for next round
			totalTime = 0.0;
			frameCount = 0;
		}


		cv::Mat rgb;

		if (m_filterType == FILTER_NONE)
		{
			cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
		}
		else
		{
			cv::cvtColor(result, rgb, cv::COLOR_BGR2RGB);
		}

		// Convert to CImage for MFC display
		CImage image;
		image.Create(rgb.cols, rgb.rows, 24);

		for (int y = 0; y < rgb.rows; y++) {
			BYTE* dst = (BYTE*)image.GetPixelAddress(0, y);
			BYTE* src = rgb.ptr(y);
			memcpy(dst, src, rgb.cols * 3);
		}

		// Draw into the picture control
		CClientDC dc(&m_imageDisplay);
		image.Draw(dc, 0, 0);
	}

	// Optionally call base class

	CString mode = m_useGPU ? _T("GPU") : _T("CPU");
	CString filter;

	switch (m_filterType) {
	case FILTER_NONE:  filter = _T("None"); break;
	case FILTER_EDGE: filter = _T("Edge"); break;
	case FILTER_BLUR:  filter = _T("Blur"); break;
	}

	//CString title;
	//title.Format(_T("Filter: %s | Mode: %s | Avg time: %.2f ms (over %d frames)"),
	//	filter, mode, avg, frameCount);

	//SetWindowText(title);


	//auto end = std::chrono::high_resolution_clock::now();
	//double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

	////  Update running average
	//totalTime += elapsed;
	//frameCount++;

	//if (frameCount >= maxFrames)
	//{
	//	double avg = totalTime / frameCount;

	//	CString title;
	//	title.Format(_T("Using %s - Avg time: %.2f ms (over %d frames)"),
	//		m_useGPU ? _T("GPU") : _T("CPU"),
	//		avg, frameCount);
	//	SetWindowText(kAppTitle + title);

	//	// Reset for next round
	//	totalTime = 0.0;
	//	frameCount = 0;
	//}


 //   if (frameCount >= maxFrames)
	//{
	//   double avg = totalTime / frameCount;

	//  CString title;
	//  title.Format(_T("Using %s - Avg time: %.2f ms (over %d frames)"),
	//	m_useGPU ? _T("GPU") : _T("CPU"),
	//	avg, frameCount);
	//  SetWindowText(title);
 //   }

	CDialogEx::OnTimer(nIDEvent);
}



cv::Mat CGPUGuruDlg::ApplyCpuBlur(const cv::Mat& input)
{
	cv::Mat output;
	cv::GaussianBlur(input, output, cv::Size(9, 9), 2.0);
	return output;
}


cv::Mat CGPUGuruDlg::ApplyCpuFilter(const cv::Mat& input)
{
	cv::Mat gray, gradX, gradY, absGradX, absGradY, output;
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
	cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);
	cv::convertScaleAbs(gradX, absGradX);
	cv::convertScaleAbs(gradY, absGradY);
	cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, output);
	cv::cvtColor(output, output, cv::COLOR_GRAY2RGB); // match original size
	return output;
}


// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGPUGuruDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}





void CGPUGuruDlg::OnBnClickedCbGpu()
{
	totalTime = 0.0;
	frameCount = 0;
	SetWindowText( kAppTitle + _T("GPU mode changed — timing reset"));
}

void CGPUGuruDlg::OnCbnSelchangeCbFilter()
{
	totalTime = 0.0;
	frameCount = 0;
	SetWindowText(kAppTitle +_T("Filter changed — timing reset"));
}
