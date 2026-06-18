"""Tests for Pascal-VOC -> YOLO conversion."""

from qc.convert import voc_xml_to_yolo_lines

XML = """<annotation>
  <size><width>200</width><height>200</height></size>
  <object><name>scratches</name><bndbox>
    <xmin>50</xmin><ymin>50</ymin><xmax>150</xmax><ymax>150</ymax></bndbox></object>
  <object><name>crazing</name><bndbox>
    <xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>200</ymax></bndbox></object>
</annotation>"""


def test_two_objects_converted():
    lines = voc_xml_to_yolo_lines(XML, {"crazing": 0, "scratches": 1})
    assert len(lines) == 2
    # scratches box is centred and half-size
    assert lines[0].startswith("1 0.500000 0.500000 0.500000 0.500000")


def test_unknown_class_is_skipped():
    lines = voc_xml_to_yolo_lines(XML, {"scratches": 0})  # 'crazing' not in the map
    assert len(lines) == 1 and lines[0].startswith("0 ")


def test_default_size_used_when_missing():
    xml = "<annotation><object><name>a</name><bndbox><xmin>0</xmin><ymin>0</ymin>" \
          "<xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>"
    lines = voc_xml_to_yolo_lines(xml, {"a": 0}, default_size=(20, 20))
    assert lines[0].startswith("0 0.250000 0.250000 0.500000 0.500000")
