# page_id: page_eventbrite_39adaf730c584c5582b89d1335e0c2cd_05
# screenshot: 2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7.png
# step_index: 5/6
# task: Open Eventbrite. Search for 'food and drink' events. Follow the organizer of the first event in listing.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas.
# Variables provided by the environment:
# - canvas: PIL.Image (RGB) of size 1440x2960
# - draw: PIL.ImageDraw instance attached to canvas
# - font_sm, font_md, font_lg, font_xl  (unused here)

W, H = canvas.size

# 1) Overall page background (very subtle off-white to match screenshot)
draw.rectangle((0, 0, W, H), fill="#FBFBFD")

# 2) Status bar area at top (~50-72px high)
status_h = 72
draw.rectangle((0, 0, W, status_h), fill="#D0D0D0")  # light gray status bar

# subtle bottom divider for status bar
draw.line((0, status_h - 1, W, status_h - 1), fill="#BDBDBD", width=1)

# 3) Top image area placeholder background (we do NOT draw the image itself).
# Provide a neutral dark gradient-like strip behind where the image will be placed
# to simulate the darkened header background edges.
img_top_h = 420
# Left and right subtle vignette strips
vignette_w = 140
draw.rectangle((0, status_h, vignette_w, img_top_h), fill="#F2F2F4")
draw.rectangle((W - vignette_w, status_h, W, img_top_h), fill="#F2F2F4")
# A faint horizontal overlay/progress track near the lower part of the image
prog_y1 = status_h + 320
prog_y2 = prog_y1 + 12
draw.rectangle((48, prog_y1, W - 48, prog_y2), fill="#E6E6E6", outline=None)
# progress indicator (lighter segment)
draw.rectangle((48, prog_y1, int(48 + (W - 96) * 0.35), prog_y2), fill="#FFFFFF")

# 4) Yellow announcement/banner (rounded) under the header image
banner_x1, banner_x2 = 40, W - 40
banner_y1, banner_y2 = img_top_h + 12, img_top_h + 92
draw.rounded_rectangle((banner_x1, banner_y1, banner_x2, banner_y2),
                       radius=16, fill="#FFF3A8", outline="#F0E38F", width=1)

# 5) Main content card for organizer (rounded light card behind profile & follow)
card_x1, card_x2 = 40, W - 40
card_y1, card_y2 = banner_y2 + 28, banner_y2 + 160
draw.rounded_rectangle((card_x1, card_y1, card_x2, card_y2),
                       radius=20, fill="#F7F7FB", outline="#E6E6EA", width=1)

# subtle inner divider line on the organizer card (to suggest separation)
divider_y = card_y1 + 94
draw.line((card_x1 + 20, divider_y, card_x2 - 20, divider_y), fill="#ECEBF0", width=1)

# 6) Thin separators between major sections
sep_y_1 = card_y2 + 40
draw.line((40, sep_y_1, W - 40, sep_y_1), fill="#ECEBF0", width=1)

# Another separator under policy text area
sep_y_2 = sep_y_1 + 220
draw.line((40, sep_y_2, W - 40, sep_y_2), fill="#F0EFF3", width=1)

# 7) Date/time selection container background (rounded card behind date tiles)
dates_x1, dates_x2 = 24, W - 24
dates_y1, dates_y2 = sep_y_1 + 36, sep_y_1 + 420
draw.rounded_rectangle((dates_x1, dates_y1, dates_x2, dates_y2),
                       radius=18, fill="#FFFFFF", outline="#F0EDF3", width=1)

# Soft drop shadow under the dates container (a faint gray strip)
shadow_y1 = dates_y2
draw.rectangle((dates_x1 + 2, shadow_y1, dates_x2 - 2, shadow_y1 + 8), fill="#F4F4F6")

# 8) Bottom action bar background (fixed bottom area with light gray fill)
bottom_h = 280
bottom_y1 = H - bottom_h
draw.rectangle((0, bottom_y1, W, H), fill="#F6F5F8")
# top divider for bottom bar
draw.line((24, bottom_y1 + 8, W - 24, bottom_y1 + 8), fill="#E6E5E9", width=1)

# subtle rounded inner panel to hold buttons (background only)
panel_x1, panel_x2 = 40, W - 40
panel_y1, panel_y2 = bottom_y1 + 28, H - 40
draw.rounded_rectangle((panel_x1, panel_y1, panel_x2, panel_y2),
                       radius=10, fill="#FFFFFF", outline="#DAD9DE", width=1)

# 9) Final subtle horizontal guide lines for layout rhythm (do not resemble text/icons)
# These are purely structural separators at various places used by the UI.
guide_lines = [
    card_y1 - 14,
    card_y2 + 10,
    dates_y1 - 18,
    dates_y2 + 10,
    bottom_y1 + 4
]
for y in guide_lines:
    draw.line((36, y, W - 36, y), fill="#F3F2F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1236), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1236, 1344, 1380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/01_icon_Details.png
try:
    _c1 = get_crop(1, 522, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Details"] = [822, 2768, 1344, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/03_icon_23.png
try:
    _c3 = get_crop(3, 450, 516)
    canvas.paste(_c3, (24, 2140), _c3)
except Exception:
    pass
layout["23"] = [24, 2140, 474, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/05_icon_7.44_my.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (36, 108), _c5)
except Exception:
    pass
layout["7.44_my"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/06_icon_Sales_ended.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1116, 108), _c6)
except Exception:
    pass
layout["Sales_ended"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/07_icon_25.png
try:
    _c7 = get_crop(7, 450, 516)
    canvas.paste(_c7, (924, 2140), _c7)
except Exception:
    pass
layout["25"] = [924, 2140, 1374, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/08_icon_24.png
try:
    _c8 = get_crop(8, 450, 516)
    canvas.paste(_c8, (474, 2140), _c8)
except Exception:
    pass
layout["24"] = [474, 2140, 924, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/09_icon_5.30_PM.png
try:
    _c9 = get_crop(9, 291, 144)
    canvas.paste(_c9, (288, 1196), _c9)
except Exception:
    pass
layout["5.30_PM"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 98, 66)
    canvas.paste(_c10, (1212, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/11_icon_7.44_my.png
try:
    _c11 = get_crop(11, 62, 70)
    canvas.paste(_c11, (180, 0), _c11)
except Exception:
    pass
layout["7.44_my"] = [180, 0, 242, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/12_icon_5.30_PM.png
try:
    _c12 = get_crop(12, 291, 144)
    canvas.paste(_c12, (288, 1196), _c12)
except Exception:
    pass
layout["5.30_PM"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 60, 66)
    canvas.paste(_c13, (1315, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1315, 0, 1375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 69, 72)
    canvas.paste(_c14, (307, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [307, 1, 376, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 58, 70)
    canvas.paste(_c15, (246, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [246, 1, 304, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 73)
    canvas.paste(_c16, (381, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [381, 0, 433, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/17_text_7.44_my.png
try:
    _c17 = get_crop(17, 149, 43)
    canvas.paste(_c17, (22, 15), _c17)
except Exception:
    pass
layout["7.44_my"] = [22, 15, 171, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/18_text_7_KINGDOMS.png
try:
    _c18 = get_crop(18, 291, 144)
    canvas.paste(_c18, (288, 1196), _c18)
except Exception:
    pass
layout["7_KINGDOMS"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/19_text_NGDOM.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (96, 1235), _c19)
except Exception:
    pass
layout["NGDOM"] = [96, 1235, 240, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/20_text_365_Followers.png
try:
    _c20 = get_crop(20, 291, 144)
    canvas.paste(_c20, (288, 1196), _c20)
except Exception:
    pass
layout["365_Followers"] = [288, 1196, 579, 1340]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/21_text_Kingdoms.png
try:
    _c21 = get_crop(21, 227, 66)
    canvas.paste(_c21, (174, 1509), _c21)
except Exception:
    pass
layout["Kingdoms"] = [174, 1509, 401, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/22_text_2_hrs.png
try:
    _c22 = get_crop(22, 112, 50)
    canvas.paste(_c22, (141, 1621), _c22)
except Exception:
    pass
layout["2_hrs"] = [141, 1621, 253, 1671]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 60)
    canvas.paste(_c23, (138, 1727), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1727, 437, 1787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/24_text_The_organizer_will_review_refund_request.png
try:
    _c24 = get_crop(24, 1344, 144)
    canvas.paste(_c24, (48, 1463), _c24)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 1463, 1392, 1607]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/25_text_Select_date_and_time.png
try:
    _c25 = get_crop(25, 450, 516)
    canvas.paste(_c25, (24, 2140), _c25)
except Exception:
    pass
layout["Select_date_and_time"] = [24, 2140, 474, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/26_text_Thursday.png
try:
    _c26 = get_crop(26, 191, 63)
    canvas.paste(_c26, (1052, 2211), _c26)
except Exception:
    pass
layout["Thursday"] = [1052, 2211, 1243, 2274]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/27_text_April.png
try:
    _c27 = get_crop(27, 450, 516)
    canvas.paste(_c27, (924, 2140), _c27)
except Exception:
    pass
layout["April"] = [924, 2140, 1374, 2656]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_05_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-7/28_text_Sales_ended.png
try:
    _c28 = get_crop(28, 274, 55)
    canvas.paste(_c28, (90, 2814), _c28)
except Exception:
    pass
layout["Sales_ended"] = [90, 2814, 364, 2869]
