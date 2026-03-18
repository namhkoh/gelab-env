# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_03
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5.png
# step_index: 3/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill full background (dominant color = white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top ~56px) - light gray bar
status_bar_h = 56
draw.rectangle((0, 0, 1440, status_bar_h), fill=(190, 190, 190))

# Subtle separator/shadow under status bar
draw.rectangle((0, status_bar_h, 1440, status_bar_h + 3), fill=(220, 220, 220))

# Header area (search/title area) sits under status bar.
# Draw a faint white toolbar area (canvas already white) and a strong blue underline
header_top = status_bar_h
header_bottom = 140
# Blue underline (thick)
underline_h = 6
draw.rectangle((48, header_bottom - underline_h, 1440 - 48, header_bottom), fill=(29, 78, 255))

# Light divider just above content area
draw.rectangle((0, header_bottom, 1440, header_bottom + 1), fill=(240, 240, 240))

# "Popular" and quick filters area background: keep white but add faint left rule to separate nav margin
# (no text or icons drawn)

# Events section background panel (subtle off-white to distinguish from page background)
events_panel_top = 1000
events_panel_left = 48
events_panel_right = 1440 - 48
events_panel_bottom = 2760
draw.rounded_rectangle(
    (events_panel_left, events_panel_top, events_panel_right, events_panel_bottom),
    radius=8,
    fill=(250, 250, 251),
    outline=None
)

# Draw event row card backgrounds with slight shadow/highlight to create structure.
# Use the detected event row coordinates (draw only background rounded cards)
event_rows = [
    (48, 1117, 48 + 1344, 1117 + 396),
    (48, 1513, 48 + 1344, 1513 + 396),
    (48, 1909, 48 + 1344, 1909 + 396),
    (48, 2305, 48 + 1344, 2305 + 396),
]
for (l, t, r, b) in event_rows:
    # subtle drop shadow
    shadow_rect = (l + 6, t + 6, r + 6, b + 6)
    draw.rounded_rectangle(shadow_rect, radius=8, fill=(240, 240, 242))
    # white card
    draw.rounded_rectangle((l, t, r, b), radius=8, fill=(255, 255, 255))

    # thin separator line at bottom of each card (soft gray)
    draw.line((l + 8, b, r - 8, b), fill=(235, 235, 238), width=1)

# Separators between major sections above events (between filters and events)
# Light horizontal divider
draw.line((48, 980, 1440 - 48, 980), fill=(230, 230, 235), width=1)

# Additional subtle vertical guides/margins (left content margin)
draw.rectangle((0, events_panel_top, 48, events_panel_bottom), fill=(255, 255, 255))
draw.rectangle((1440 - 48, events_panel_top, 1440, events_panel_bottom), fill=(255, 255, 255))

# Bottom navigation bar background (area at y ~2804..2960)
nav_top = 2804
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill=(255, 255, 255))
# Top border for nav bar
draw.rectangle((0, nav_top, 1440, nav_top + 1), fill=(230, 230, 234))
# Slight elevation highlight for active center slot (do not draw icons)
center_icon_x = 720
center_highlight_radius = 28
draw.ellipse((center_icon_x - center_highlight_radius, nav_top - 8,
              center_icon_x + center_highlight_radius, nav_top + center_highlight_radius),
             fill=(255, 255, 255, 0), outline=(230, 112, 40))

# Small subtle bottom shadow to ground the nav bar
draw.rectangle((0, nav_bottom - 2, 1440, nav_bottom), fill=(245, 245, 245))

# Final soft vignette on content area edges (very subtle)
edge_strip = 12
draw.rectangle((0, header_bottom, edge_strip, events_panel_bottom), fill=(255, 255, 255))
draw.rectangle((1440 - edge_strip, header_bottom, 1440, events_panel_bottom), fill=(255, 255, 255))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/00_icon_he.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["[he"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/01_icon_MUSIC.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1117), _c1)
except Exception:
    pass
layout["MUSIC"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/02_icon_for_a_fun_per.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1513), _c2)
except Exception:
    pass
layout["for_a_fun_per"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/03_icon_6.48.png
try:
    _c3 = get_crop(3, 133, 112)
    canvas.paste(_c3, (49, 114), _c3)
except Exception:
    pass
layout["6.48"] = [49, 114, 182, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/04_icon_Music.png
try:
    _c4 = get_crop(4, 60, 62)
    canvas.paste(_c4, (311, 2), _c4)
except Exception:
    pass
layout["Music"] = [311, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/05_icon_6.48.png
try:
    _c5 = get_crop(5, 55, 62)
    canvas.paste(_c5, (182, 2), _c5)
except Exception:
    pass
layout["6.48"] = [182, 2, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/06_icon_6.48.png
try:
    _c6 = get_crop(6, 58, 64)
    canvas.paste(_c6, (115, 1), _c6)
except Exception:
    pass
layout["6.48"] = [115, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/07_icon_Music.png
try:
    _c7 = get_crop(7, 1344, 191)
    canvas.paste(_c7, (48, 72), _c7)
except Exception:
    pass
layout["Music"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/08_icon_Music.png
try:
    _c8 = get_crop(8, 46, 59)
    canvas.paste(_c8, (251, 4), _c8)
except Exception:
    pass
layout["Music"] = [251, 4, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/09_icon_Cr.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Cr"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/10_icon_Roulette.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1909), _c10)
except Exception:
    pass
layout["Roulette"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/11_icon_Carteret_Public_Library.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1513), _c11)
except Exception:
    pass
layout["Carteret_Public_Library"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/12_icon_Greenwich_House_Music_School.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1117), _c12)
except Exception:
    pass
layout["Greenwich_House_Music_Sch"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/13_icon_Cancel.png
try:
    _c13 = get_crop(13, 99, 63)
    canvas.paste(_c13, (1213, 0), _c13)
except Exception:
    pass
layout["Cancel"] = [1213, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 50, 62)
    canvas.paste(_c14, (1321, 1), _c14)
except Exception:
    pass
layout["Cancel"] = [1321, 1, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1099, 96), _c15)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/16_icon_Rum_and_Music.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 2305), _c16)
except Exception:
    pass
layout["Rum_and_Music"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/17_icon_8_2153_creator_followers.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["8_2153_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/18_icon_Generate_Music.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1909), _c18)
except Exception:
    pass
layout["Generate_Music"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/19_icon_Generate_Music.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1909), _c19)
except Exception:
    pass
layout["Generate_Music"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/20_icon_Rockness_Music.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1513), _c20)
except Exception:
    pass
layout["Rockness_Music"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/21_icon_More.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (1152, 2804), _c21)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/22_icon_Pier_4.png
try:
    _c22 = get_crop(22, 100, 52)
    canvas.paste(_c22, (391, 2542), _c22)
except Exception:
    pass
layout["Pier_4"] = [391, 2542, 491, 2594]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/23_icon_7_00_PM_EDT.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1909), _c23)
except Exception:
    pass
layout["7:00_PM_EDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/24_icon_house_music.png
try:
    _c24 = get_crop(24, 1344, 120)
    canvas.paste(_c24, (48, 618), _c24)
except Exception:
    pass
layout["house_music"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/26_icon_6.48.png
try:
    _c26 = get_crop(26, 91, 61)
    canvas.paste(_c26, (16, 3), _c26)
except Exception:
    pass
layout["6.48"] = [16, 3, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/27_icon_music_concert.png
try:
    _c27 = get_crop(27, 90, 95)
    canvas.paste(_c27, (37, 767), _c27)
except Exception:
    pass
layout["music_concert"] = [37, 767, 127, 862]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/28_icon_Roulette.png
try:
    _c28 = get_crop(28, 143, 54)
    canvas.paste(_c28, (391, 2113), _c28)
except Exception:
    pass
layout["Roulette"] = [391, 2113, 534, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/29_icon_4_00_PM_EDT.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1513), _c29)
except Exception:
    pass
layout["4:00_PM_EDT"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/30_icon_Cancel.png
try:
    _c30 = get_crop(30, 149, 144)
    canvas.paste(_c30, (1243, 97), _c30)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/31_icon_techno_music.png
try:
    _c31 = get_crop(31, 95, 93)
    canvas.paste(_c31, (31, 530), _c31)
except Exception:
    pass
layout["techno_music"] = [31, 530, 126, 623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/32_icon_music_concert.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 738), _c32)
except Exception:
    pass
layout["music_concert"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/33_text_Popular.png
try:
    _c33 = get_crop(33, 221, 78)
    canvas.paste(_c33, (44, 298), _c33)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/34_text_live_music.png
try:
    _c34 = get_crop(34, 193, 48)
    canvas.paste(_c34, (162, 430), _c34)
except Exception:
    pass
layout["live_music"] = [162, 430, 355, 478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/35_text_techno_music.png
try:
    _c35 = get_crop(35, 258, 43)
    canvas.paste(_c35, (163, 552), _c35)
except Exception:
    pass
layout["techno_music"] = [163, 552, 421, 595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/36_text_classical_music.png
try:
    _c36 = get_crop(36, 290, 45)
    canvas.paste(_c36, (161, 910), _c36)
except Exception:
    pass
layout["classical_music"] = [161, 910, 451, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/37_text_Events.png
try:
    _c37 = get_crop(37, 188, 61)
    canvas.paste(_c37, (45, 1026), _c37)
except Exception:
    pass
layout["Events"] = [45, 1026, 233, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/38_text_Cr.png
try:
    _c38 = get_crop(38, 37, 12)
    canvas.paste(_c38, (773, 2794), _c38)
except Exception:
    pass
layout["Cr"] = [773, 2794, 810, 2806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/39_clickable_live_music.png
try:
    _c39 = get_crop(39, 1344, 120)
    canvas.paste(_c39, (48, 378), _c39)
except Exception:
    pass
layout["live_music"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/40_clickable_techno_music.png
try:
    _c40 = get_crop(40, 1344, 120)
    canvas.paste(_c40, (48, 498), _c40)
except Exception:
    pass
layout["techno_music"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/41_clickable_classical_music.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 858), _c41)
except Exception:
    pass
layout["classical_music"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_03_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-5/42_clickable_Favorites.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (576, 2804), _c42)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
