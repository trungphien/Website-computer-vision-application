import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="🏠",
)

st.image('images\\title.jpg')

# Inject CSS với styling cho content box và headings
st.markdown(
    """
    <style>
    .content-box {{
        background: rgba(0, 0, 0, 0.4); /* Lighter semi-transparent black */
        padding: 20px;
        border-radius: 10px;
        margin: 20px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        color: white; /* Set text color to white */
    }}
    h3 {{
        color: #FFFFFF; /* Bright white color for h3 */
    }}
    a {{
        color: #1E90FF; /* Optional: Adjust link color */
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Content within the box
st.markdown(
    """
    <div class="content-box">
        <h3>Website Xử Lý Ảnh Số</h3>
        <ul>
            <li>Thực hiện bởi: Nguyễn Trung Phiên và Hoàng Mai Hiếu</li>
            <li>Giảng viên hướng dẫn: ThS. Trần Tiến Đức</li>
            <li>Lớp Xử Lý Ảnh Số nhóm 01: DIPR430685_23_1_01</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="content-box">
        <h3>Thông tin liên hệ</h3>
        <ul>
            <li>Facebook: <a href="https://www.facebook.com/phien.phassphachss">Nguyễn Trung Phiên</a> hoặc <a href="https://www.facebook.com/tuilahiuu">Hoàng Mai Hiếu</a></li>
            <li>Email: 21110593@student.hcmute.edu.vn hoặc 21110882@student.hcmute.edu.vn</li>
            <li>Lấy source code tại <a href="https://github.com/quangnghia1110/XuLyAnhSo.git">đây</a></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="content-box">
        <h3>Video giới thiệu về Website</h3>
        <a href="#">Video giới thiệu website Xử Lý Ảnh Số</a>
    </div>
    """,
    unsafe_allow_html=True
)
