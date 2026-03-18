# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_18
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20.png
# step_index: 18/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. This script paints the page background, status bar,
# header/toolbar backgrounds, section card backgrounds (rounded), separators, and bottom nav bar.
# Do not draw any icons/text/content that will be pasted on top.

# Color palette
bg_color = (249, 250, 251)        # overall very light gray background
status_bar_color = (189, 189, 189)  # muted gray for status bar
header_bg = (255, 255, 255)       # white header/toolbar
divider_color = (226, 228, 230)   # light divider lines
card_bg = (255, 255, 255)         # card white
card_shadow = (235, 238, 240)     # subtle shadow/backdrop for cards
bottom_nav_bg = (255, 255, 255)   # bottom nav background (white)
accent_strip = (244, 246, 247)    # very subtle strip used for larger content areas

w, h = canvas.size

# 1) Full background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# 2) Status bar (approx ~50px tall; using 80px to match screenshot spacing)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Thin subtle line under status bar to separate from header
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# 3) Header / Search area background
# Header area extends below status bar; leave space for search field and filter chips.
header_h = 200
header_top = status_h
header_bottom = header_top + header_h
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# A subtle horizontal divider under header
draw.line([(36, header_bottom), (w-36, header_bottom)], fill=divider_color, width=1)

# 4) Content area - draw a faint full-width strip where filters/controls sit (but do not draw the actual buttons)
filters_strip_top = header_bottom + 18
filters_strip_bottom = filters_strip_top + 72
draw.rectangle([(0, filters_strip_top), (w, filters_strip_bottom)], fill=accent_strip)
draw.line([(36, filters_strip_bottom), (w-36, filters_strip_bottom)], fill=divider_color, width=1)

# 5) Event cards - draw background rounded cards with subtle shadow/backdrop.
# Card layout approximations (positions chosen to sit behind detected elements)
card_x = 48
card_w = w - 2*card_x

# First event card (top)
card1_y = filters_strip_bottom + 36
card1_h = 560
# shadow/backdrop
draw.rounded_rectangle(
    [(card_x, card1_y+12), (card_x+card_w, card1_y+12+card1_h)],
    radius=20,
    fill=card_shadow
)
# card foreground
draw.rounded_rectangle(
    [(card_x, card1_y), (card_x+card_w, card1_y+card1_h)],
    radius=20,
    fill=card_bg
)
# thin divider line below first card content area (separates image area from meta area)
draw.line([(card_x+28, card1_y+card1_h-160), (card_x+card_w-28, card1_y+card1_h-160)], fill=divider_color, width=1)

# Second event card
card2_y = card1_y + card1_h + 48
card2_h = 520
draw.rounded_rectangle(
    [(card_x, card2_y+12), (card_x+card_w, card2_y+12+card2_h)],
    radius=20,
    fill=card_shadow
)
draw.rounded_rectangle(
    [(card_x, card2_y), (card_x+card_w, card2_y+card2_h)],
    radius=20,
    fill=card_bg
)
draw.line([(card_x+28, card2_y+card2_h-160), (card_x+card_w-28, card2_y+card2_h-160)], fill=divider_color, width=1)

# 6) Additional content band behind lower listing (a very light band to separate lists)
list_band_top = card2_y + card2_h + 36
list_band_bottom = list_band_top + 36
draw.rectangle([(0, list_band_top), (w, list_band_bottom)], fill=accent_strip)

# 7) Subtle separators between list items (full width thin lines)
sep_y = list_band_bottom + 220
while sep_y < h - 220:
    draw.line([(36, sep_y), (w-36, sep_y)], fill=divider_color, width=1)
    sep_y += 220

# 8) Bottom navigation bar
bottom_h = 120
bottom_top = h - bottom_h
# top divider for nav
draw.line([(0, bottom_top), (w, bottom_top)], fill=divider_color, width=1)
draw.rectangle([(0, bottom_top), (w, h)], fill=bottom_nav_bg)

# 9) Floating subtle top/bottom padding bars to visually balance page edges
edge_strip = 18
draw.rectangle([(0, header_top+edge_strip), (w, header_top+edge_strip+2)], fill=divider_color)
draw.rectangle([(0, bottom_top-edge_strip-2), (w, bottom_top-edge_strip)], fill=divider_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/00_icon_15_2024.png
try:
    _c0 = get_crop(0, 584, 135)
    canvas.paste(_c0, (458, 390), _c0)
except Exception:
    pass
layout["15,2024"] = [458, 390, 1042, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (1054, 390), _c1)
except Exception:
    pass
layout["Music"] = [1054, 390, 1241, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 392, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["2_Filters"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/03_icon_Tou.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Tou"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/04_icon_MDEF.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2331), _c4)
except Exception:
    pass
layout["MDEF"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/05_icon_938-9878.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["938-9878"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/06_icon_MDEF.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2331), _c6)
except Exception:
    pass
layout["MDEF"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/07_icon_Bass.png
try:
    _c7 = get_crop(7, 1344, 1001)
    canvas.paste(_c7, (48, 1815), _c7)
except Exception:
    pass
layout["Bass"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 66)
    canvas.paste(_c8, (1152, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1152, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/09_icon_7.48.png
try:
    _c9 = get_crop(9, 119, 114)
    canvas.paste(_c9, (58, 114), _c9)
except Exception:
    pass
layout["7.48"] = [58, 114, 177, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 65, 63)
    canvas.paste(_c10, (308, 1), _c10)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/11_icon_7.48.png
try:
    _c11 = get_crop(11, 58, 65)
    canvas.paste(_c11, (115, 0), _c11)
except Exception:
    pass
layout["7.48"] = [115, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/12_icon_7.48.png
try:
    _c12 = get_crop(12, 60, 65)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["7.48"] = [180, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 64, 63)
    canvas.paste(_c13, (1213, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1213, 0, 1277, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 63)
    canvas.paste(_c14, (247, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 1, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 62)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/16_icon_2_._9_00_PM_PDT.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["2_._9:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/17_icon_INFO.png
try:
    _c17 = get_crop(17, 1344, 1091)
    canvas.paste(_c17, (48, 676), _c17)
except Exception:
    pass
layout["INFO"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 49, 63)
    canvas.paste(_c18, (383, 1), _c18)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/20_icon_Thu.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Thu,"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/21_icon_Tou.png
try:
    _c21 = get_crop(21, 50, 73)
    canvas.paste(_c21, (1093, 1368), _c21)
except Exception:
    pass
layout["Tou"] = [1093, 1368, 1143, 1441]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/22_icon_San_Francisco.png
try:
    _c22 = get_crop(22, 536, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/23_icon_Super_Bass_Hip_Hop_Thursdays_Party_at_Be.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Super_Bass_Hip_Hop_Thursd"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 43, 63)
    canvas.paste(_c24, (1272, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1272, 0, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/25_icon_Bollywood_Takeover_One_Last_Dance.png
try:
    _c25 = get_crop(25, 1344, 1091)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["Bollywood_Takeover:_One_L"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/26_icon_Super_Bass_Hip_Hop_Thursdays_Party_at_Be.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Super_Bass_Hip_Hop_Thursd"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/27_icon_7.48.png
try:
    _c27 = get_crop(27, 94, 65)
    canvas.paste(_c27, (12, 0), _c27)
except Exception:
    pass
layout["7.48"] = [12, 0, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/28_icon_Temple_Nightclub_San_Francisco.png
try:
    _c28 = get_crop(28, 41, 60)
    canvas.paste(_c28, (286, 1662), _c28)
except Exception:
    pass
layout["Temple_Nightclub_San_Fran"] = [286, 1662, 327, 1722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/30_icon_Free.png
try:
    _c30 = get_crop(30, 128, 77)
    canvas.paste(_c30, (91, 2508), _c30)
except Exception:
    pass
layout["Free"] = [91, 2508, 219, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/31_text_479_events.png
try:
    _c31 = get_crop(31, 392, 135)
    canvas.paste(_c31, (54, 390), _c31)
except Exception:
    pass
layout["479_events"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/32_text_BIH.png
try:
    _c32 = get_crop(32, 132, 86)
    canvas.paste(_c32, (90, 714), _c32)
except Exception:
    pass
layout["BIH"] = [90, 714, 222, 800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/33_text_FRIDAY.png
try:
    _c33 = get_crop(33, 142, 55)
    canvas.paste(_c33, (88, 853), _c33)
except Exception:
    pass
layout["FRIDAY"] = [88, 853, 230, 908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/34_text_Temple_Nightclub_San_Francisco.png
try:
    _c34 = get_crop(34, 600, 57)
    canvas.paste(_c34, (90, 1600), _c34)
except Exception:
    pass
layout["Temple_Nightclub_San_Fran"] = [90, 1600, 690, 1657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/35_text_Thu.png
try:
    _c35 = get_crop(35, 91, 48)
    canvas.paste(_c35, (93, 2761), _c35)
except Exception:
    pass
layout["Thu,"] = [93, 2761, 184, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/36_text_2_._9_00_PM_PDT.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (288, 2804), _c36)
except Exception:
    pass
layout["2_._9:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/37_text_DANCE.png
try:
    _c37 = get_crop(37, 272, 94)
    canvas.paste(_c37, (993, 713), _c37)
except Exception:
    pass
layout["DANCE'"] = [993, 713, 1265, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/38_text_Last.png
try:
    _c38 = get_crop(38, 200, 79)
    canvas.paste(_c38, (787, 761), _c38)
except Exception:
    pass
layout["Last"] = [787, 761, 987, 840]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_18_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-20/39_text_ONE.png
try:
    _c39 = get_crop(39, 196, 79)
    canvas.paste(_c39, (587, 795), _c39)
except Exception:
    pass
layout["'ONE"] = [587, 795, 783, 874]
