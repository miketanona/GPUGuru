
// GPUGuruDlg.h : header file
//

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>


// CGPUGuruDlg dialog
class CGPUGuruDlg : public CDialogEx
{
// Construction
public:
	CGPUGuruDlg(CWnd* pParent = nullptr);	// standard constructor

	enum FilterType {
		FILTER_NONE,
		FILTER_EDGE,
		FILTER_BLUR
	};


	FilterType m_filterType = FILTER_NONE;

	int frameCount = 0;
	int avg = 0;
	double totalTime = 0.0;
	const int maxFrames = 60;  // Average over last 60 frames


// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_GPUGURU_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;
	BOOL m_useGPU;
	CStatic m_imageDisplay; // your picture control
	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

private:
	//CStatic m_imageDisplay;
	cv::VideoCapture cap;
	cv::Mat frame;
	void ResizeWindow();
	void ResizeDialogToFitCamera();
	BOOL OnEraseBkgnd(CDC* pDC);
	cv::Mat ApplyCpuBlur(const cv::Mat& input);

public:
	cv::Mat ApplyCpuFilter(const cv::Mat& input);

	afx_msg void OnBnClickedCbGpu();
	afx_msg void OnCbnSelchangeCbFilter();
};
