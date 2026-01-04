from copy import deepcopy
from typing import Tuple, Optional, Dict, NoReturn, List


class ClassMapper:

    # RGB color values
    CODE2ALIAS = dict(
        AVICOMP={
            "AVICOMP00": ["Background"],
            "AVICOMP01": ["SolderMask"],
            "AVICOMP02": ["SurfaceFinish"],
            "AVICOMP03": ["ViaCircle"],
            "AVICOMP04": ["Via"],
            "AVICOMP05": ["Silk"],
            "AVICOMP06": ["Laser", "LaserChar"],
            "AVICOMP07": ["VCut"],
            "AVICOMP08": ["SilkChar"],
            "AVICOMP09": ["Circuit"],
            "AVICOMP10": ["GoldChar"],
            "AVICOMP11": ["Finger"],
            "AVICOMP12": ["OpticalPoint"],
            "AVICOMP13": ["ScreenFoolProof"],
            "AVICOMP14": ["Special"],
            "AVICOMP15": ["TieBar"],
            "AVICOMP16": ["QRCode"],
        },
        AVICOMPData={
            "AVICOMPData00": ["Background"],
            "AVICOMPData01": ["SolderMask", "Solder Mask"],
            "AVICOMPData02": ["SurfaceFinish", "Surface Finish"],
            "AVICOMPData03": ["ViaCircle", "Via Circle"],
            "AVICOMPData04": ["Via"],
            "AVICOMPData05": ["Silk"],
            "AVICOMPData06": ["Laser", "LaserChar", "Laser char"],
            "AVICOMPData07": ["VCut"],
            "AVICOMPData08": ["SilkChar", "Silk Char"],
            "AVICOMPData09": ["Circuit"],
            "AVICOMPData10": ["GoldChar", "Gold char"],
            "AVICOMPData11": ["Finger"],
            "AVICOMPData12": ["OpticalPoint", "Optical Point"],
            "AVICOMPData13": ["ScreenFoolProof"],
            "AVICOMPData14": ["Special"],
            "AVICOMPData15": ["TieBar", "Tie Bar"],
            "AVICOMPData16": ["QRCode", "QR Code"],
            "AVICOMPData17": ["BigCopper", "dtm"],
            "AVICOMPData18": ["SemiSlot"],
            "AVICOMPData19": ["UnderMaskChar", "ymx Char", "ymx"],
            "AVICOMPData20": ["SurfaceOri"],
        },
        AOICOMP={
            "AOICOMP00": ["Substrate"],
            "AOICOMP01": ["Circuit"],
            "AOICOMP02": ["Pad"],
            "AOICOMP03": ["CopperSurface", "Coppersurface"],
            "AOICOMP04": ["ViaCircle", "Viacircle"],
            "AOICOMP05": ["Via"],
            "AOICOMP06": ["MetalChar", "Metal_char"],
            "AOICOMP07": ["SubChar", "Sub_char"],
            "AOICOMP08": ["IsoRing", "Isoring"],
            "AOICOMP09": ["SharpAngle", "Sharp_Angle"],
            "AOICOMP10": ["CopperBridge", "Copper_Bridge"],
            "AOICOMP11": ["WindowCorner", "Window&corner"],
            "AOICOMP12": ["BackDrill"],
            "AOICOMP13": ["Resin", "POFV"],
            "AOICOMP14": ["MatterCover"],
            "AOICOMP15": ["ScreenFoolProof"],
            "AOICOMP16": ["SemiSlot"]
        },
        DEFSEG={
            "DEFSEG00": ["OK", 'Background'],
            "DEFSEG01": ["NG", 'Defect'],
        },
        AVISEMANTIC={
            "AVISEMANTIC00": ["000"],
            "AVISEMANTIC01": ["001"],
            "AVISEMANTIC02": ["0021"],
            "AVISEMANTIC03": ["0022"],
            "AVISEMANTIC04": ["0141"],
            "AVISEMANTIC05": ["0151"],
            "AVISEMANTIC06": ["0142"],
            "AVISEMANTIC07": ["0091"],
            "AVISEMANTIC08": ["0092"],
            "AVISEMANTIC09": ["0093"],
            "AVISEMANTIC10": ["0110"],
            "AVISEMANTIC11": ["0113"],
            "AVISEMANTIC12": ["062"],
            "AVISEMANTIC13": ["0112"],
            "AVISEMANTIC14": ["0111"],
            "AVISEMANTIC15": ["01112"],
            "AVISEMANTIC16": ["011"],
            "AVISEMANTIC17": ["206"],
            "AVISEMANTIC18": ["070"],
            "AVISEMANTIC19": ["207"],
            "AVISEMANTIC20": ["8087"],
            "AVISEMANTIC21": ["9275"],
            "AVISEMANTIC22": ["052"],
            "AVISEMANTIC23": ["0521"],
            "AVISEMANTIC24": ["015"],
            "AVISEMANTIC25": ["808"],
            "AVISEMANTIC26": ["0140"],
            "AVISEMANTIC27": ["014"],
            "AVISEMANTIC28": ["012"],
            "AVISEMANTIC29": ["0120"],
            "AVISEMANTIC30": ["0129"],
            "AVISEMANTIC31": ["0128"],
            "AVISEMANTIC32": ["013"],
            "AVISEMANTIC33": ["0131"],
            "AVISEMANTIC34": ["018"],
            "AVISEMANTIC35": ["024"],
            "AVISEMANTIC36": ["017"],
            "AVISEMANTIC37": ["026"],
            "AVISEMANTIC38": ["039"],
            "AVISEMANTIC39": ["053"],
            "AVISEMANTIC40": ["054"],
            "AVISEMANTIC41": ["0341"],
            "AVISEMANTIC42": ["0342"],
            "AVISEMANTIC43": ["051"],
            "AVISEMANTIC44": ["021"],
            "AVISEMANTIC45": ["0211"],
            "AVISEMANTIC46": ["016"],
            "AVISEMANTIC47": ["505"],
            "AVISEMANTIC48": ["0421"],
            "AVISEMANTIC49": ["0420"],
            "AVISEMANTIC50": ["0419"],
            "AVISEMANTIC51": ["0422"],
            "AVISEMANTIC52": ["0221"],
            "AVISEMANTIC53": ["022"],
            "AVISEMANTIC54": ["401"],
            "AVISEMANTIC55": ["403"],
            "AVISEMANTIC56": ["404"],
            "AVISEMANTIC57": ["019"],
            "AVISEMANTIC58": ["0423"],
            "AVISEMANTIC59": ["110"],
            "AVISEMANTIC60": ["603"],
            "AVISEMANTIC61": ["405"],
            "AVISEMANTIC62": ["406"],
            "AVISEMANTIC63": ["407"],
            "AVISEMANTIC64": ["408"],
            "AVISEMANTIC65": ["409"],
            "AVISEMANTIC66": ["040"]
        },
        ICSEMANTIC={
            "ICSEMANTIC00": ["000"],
            "ICSEMANTIC01": ["001"],
            "ICSEMANTIC02": ["0021"],
            "ICSEMANTIC03": ["0022"],
            "ICSEMANTIC04": ["0141"],
            "ICSEMANTIC05": ["0151"],
            "ICSEMANTIC06": ["0142"],
            "ICSEMANTIC07": ["0091"],
            "ICSEMANTIC08": ["0092"],
            "ICSEMANTIC09": ["0093"],
            "ICSEMANTIC10": ["0110"],
            "ICSEMANTIC11": ["0113"],
            "ICSEMANTIC12": ["062"],
            "ICSEMANTIC13": ["0112"],
            "ICSEMANTIC14": ["0111"],
            "ICSEMANTIC15": ["01112"],
            "ICSEMANTIC16": ["011"],
            "ICSEMANTIC17": ["206"],
            "ICSEMANTIC18": ["070"],
            "ICSEMANTIC19": ["207"],
            "ICSEMANTIC20": ["8087"],
            "ICSEMANTIC21": ["9275"],
            "ICSEMANTIC22": ["052"],
            "ICSEMANTIC23": ["015"],
            "ICSEMANTIC24": ["808"],
            "ICSEMANTIC25": ["0140"],
            "ICSEMANTIC26": ["014"],
            "ICSEMANTIC27": ["012"],
            "ICSEMANTIC28": ["0120"],
            "ICSEMANTIC29": ["0129"],
            "ICSEMANTIC30": ["013"],
            "ICSEMANTIC31": ["0131"],
            "ICSEMANTIC32": ["018"],
            "ICSEMANTIC33": ["024"],
            "ICSEMANTIC34": ["017"],
            "ICSEMANTIC35": ["026"],
            "ICSEMANTIC36": ["039"],
            "ICSEMANTIC37": ["053"],
            "ICSEMANTIC38": ["054"],
            "ICSEMANTIC39": ["0341"],
            "ICSEMANTIC40": ["0342"],
            "ICSEMANTIC41": ["051"],
            "ICSEMANTIC42": ["021"],
            "ICSEMANTIC43": ["016"],
            "ICSEMANTIC44": ["505"],
            "ICSEMANTIC45": ["0421"],
            "ICSEMANTIC46": ["0420"],
            "ICSEMANTIC47": ["0419"],
            "ICSEMANTIC48": ["0422"],
            "ICSEMANTIC49": ["0221"],
            "ICSEMANTIC50": ["022"],
            "ICSEMANTIC51": ["401"],
            "ICSEMANTIC52": ["403"],
            "ICSEMANTIC53": ["404"],
            "ICSEMANTIC54": ["019"],
            "ICSEMANTIC55": ["0423"],
            "ICSEMANTIC56": ["603"]
        }
    )
    CODE2COLOR = dict(
        AVICOMP={
            "AVICOMP00": [(0, 0, 0)],
            "AVICOMP01": [(107, 142, 35)],
            "AVICOMP02": [(255, 255, 0), (220, 20, 60), (120, 120, 120), (0, 128, 128)],
            "AVICOMP03": [(244, 35, 232)],
            "AVICOMP04": [(0, 0, 142)],
            "AVICOMP05": [(250, 250, 250)],
            "AVICOMP06": [(173, 255, 173), (255, 153, 51)],
            "AVICOMP07": [(128, 64, 128)],
            "AVICOMP08": [(145, 44, 238)],
            "AVICOMP09": [(255, 0, 0)],
            "AVICOMP10": [(255, 222, 173)],
            "AVICOMP11": [(250, 170, 30)],
            "AVICOMP12": [(92, 172, 238)],
            "AVICOMP13": [(139, 87, 66)],
            "AVICOMP14": [(205, 133, 63)],
            "AVICOMP15": [(204, 102, 0)],
            "AVICOMP16": [(72, 209, 204)],
        },
        AVICOMPData={
            "AVICOMPData00": [(0, 0, 0)],
            "AVICOMPData01": [(107, 142, 35)],
            "AVICOMPData02": [(255, 255, 0)],
            "AVICOMPData03": [(244, 35, 232)],
            "AVICOMPData04": [(0, 0, 142)],
            "AVICOMPData05": [(250, 250, 250)],
            "AVICOMPData06": [(173, 255, 173)],
            "AVICOMPData07": [(128, 64, 128)],
            "AVICOMPData08": [(145, 44, 238)],
            "AVICOMPData09": [(255, 0, 0)],
            "AVICOMPData10": [(255, 222, 173)],
            "AVICOMPData11": [(250, 170, 30)],
            "AVICOMPData12": [(92, 172, 238)],
            "AVICOMPData13": [(139, 87, 66)],
            "AVICOMPData14": [(205, 133, 63)],
            "AVICOMPData15": [(204, 102, 0)],
            "AVICOMPData16": [(72, 209, 204)],
            "AVICOMPData17": [(120, 120, 120)],
            "AVICOMPData18": [(0, 128, 128)],
            "AVICOMPData19": [(255, 153, 51)],
            "AVICOMPData20": [(220, 20, 60)],
        },
        AOICOMP={
            "AOICOMP00": [(128, 128, 0), (0, 0, 0)],
            "AOICOMP01": [(250, 170, 30)],
            "AOICOMP02": [(220, 20, 60)],
            "AOICOMP03": [(100, 120, 250)],
            "AOICOMP04": [(244, 38, 234)],
            "AOICOMP05": [(0, 0, 145)],
            "AOICOMP06": [(255, 222, 173)],
            "AOICOMP07": [(222, 111, 0)],
            "AOICOMP08": [(111, 55, 111)],
            "AOICOMP09": [(144, 238, 144)],
            "AOICOMP10": [(0, 246, 255)],
            "AOICOMP11": [(160, 32, 240)],
            "AOICOMP12": [(250, 122, 122), (250, 120, 120)],
            "AOICOMP13": [(95, 158, 160), (100, 250, 250)],
            "AOICOMP14": [(221, 160, 221)],
            "AOICOMP15": [(0, 255, 0)],
            "AOICOMP16": [(238, 18, 137)]
        },
        DEFSEG={
            "DEFSEG00": [(0, 0, 0)],
            "DEFSEG01": [(255, 0, 0)],
        },
        AVISEMANTIC={
            "AVISEMANTIC00": [(0, 0, 0)],
            "AVISEMANTIC01": [(255, 182, 193)],
            "AVISEMANTIC02": [(220, 20, 60)],
            "AVISEMANTIC03": [(219, 112, 147)],
            "AVISEMANTIC04": [(255, 105, 180)],
            "AVISEMANTIC05": [(199, 21, 133)],
            "AVISEMANTIC06": [(218, 112, 214)],
            "AVISEMANTIC07": [(216, 191, 216)],
            "AVISEMANTIC08": [(221, 160, 221)],
            "AVISEMANTIC09": [(255, 0, 255)],
            "AVISEMANTIC10": [(128, 0, 128)],
            "AVISEMANTIC11": [(75, 0, 130)],
            "AVISEMANTIC12": [(138, 43, 226)],
            "AVISEMANTIC13": [(123, 104, 238)],
            "AVISEMANTIC14": [(230, 230, 250)],
            "AVISEMANTIC15": [(0, 0, 255)],
            "AVISEMANTIC16": [(25, 25, 112)],
            "AVISEMANTIC17": [(65, 105, 225)],
            "AVISEMANTIC18": [(176, 196, 222)],
            "AVISEMANTIC19": [(119, 136, 153)],
            "AVISEMANTIC20": [(155, 42, 42)],
            "AVISEMANTIC21": [(200, 0, 0)],
            "AVISEMANTIC22": [(30, 144, 255)],
            "AVISEMANTIC23": [(96, 60, 226)],
            "AVISEMANTIC24": [(70, 130, 180)],
            "AVISEMANTIC25": [(0, 191, 255)],
            "AVISEMANTIC26": [(95, 158, 160)],
            "AVISEMANTIC27": [(175, 238, 238)],
            "AVISEMANTIC28": [(0, 255, 255)],
            "AVISEMANTIC29": [(0, 206, 209)],
            "AVISEMANTIC30": [(0, 128, 128)],
            "AVISEMANTIC31": [(44, 22, 106)],
            "AVISEMANTIC32": [(127, 255, 170)],
            "AVISEMANTIC33": [(0, 250, 154)],
            "AVISEMANTIC34": [(0, 255, 127)],
            "AVISEMANTIC35": [(60, 179, 113)],
            "AVISEMANTIC36": [(143, 188, 143)],
            "AVISEMANTIC37": [(0, 100, 0)],
            "AVISEMANTIC38": [(124, 252, 0)],
            "AVISEMANTIC39": [(173, 255, 47)],
            "AVISEMANTIC40": [(85, 107, 47)],
            "AVISEMANTIC41": [(245, 245, 220)],
            "AVISEMANTIC42": [(190, 210, 200)],
            "AVISEMANTIC43": [(255, 255, 0)],
            "AVISEMANTIC44": [(189, 183, 107)],
            "AVISEMANTIC45": [(140, 0, 255)],
            "AVISEMANTIC46": [(255, 250, 205)],
            "AVISEMANTIC47": [(170, 150, 18)],
            "AVISEMANTIC48": [(240, 230, 140)],
            "AVISEMANTIC49": [(255, 215, 0)],
            "AVISEMANTIC50": [(255, 228, 181)],
            "AVISEMANTIC51": [(255, 165, 0)],
            "AVISEMANTIC52": [(222, 184, 135)],
            "AVISEMANTIC53": [(205, 133, 63)],
            "AVISEMANTIC54": [(255, 218, 185)],
            "AVISEMANTIC55": [(139, 69, 19)],
            "AVISEMANTIC56": [(255, 160, 122)],
            "AVISEMANTIC57": [(255, 69, 0)],
            "AVISEMANTIC58": [(255, 228, 225)],
            "AVISEMANTIC59": [(255, 150, 200)],
            "AVISEMANTIC60": [(250, 128, 114)],
            "AVISEMANTIC61": [(205, 92, 92)],
            "AVISEMANTIC62": [(255, 0, 0)],
            "AVISEMANTIC63": [(128, 0, 0)],
            "AVISEMANTIC64": [(220, 220, 220)],
            "AVISEMANTIC65": [(169, 169, 169)],
            "AVISEMANTIC66": [(0, 13, 192)],
        },
        ICSEMANTIC={
            "ICSEMANTIC00": [(0, 0, 0)],
            "ICSEMANTIC01": [(255, 182, 193)],
            "ICSEMANTIC02": [(220, 20, 60)],
            "ICSEMANTIC03": [(219, 112, 147)],
            "ICSEMANTIC04": [(255, 105, 180)],
            "ICSEMANTIC05": [(199, 21, 133)],
            "ICSEMANTIC06": [(218, 112, 214)],
            "ICSEMANTIC07": [(216, 191, 216)],
            "ICSEMANTIC08": [(221, 160, 221)],
            "ICSEMANTIC09": [(255, 0, 255)],
            "ICSEMANTIC10": [(128, 0, 128)],
            "ICSEMANTIC11": [(75, 0, 130)],
            "ICSEMANTIC12": [(138, 43, 226)],
            "ICSEMANTIC13": [(123, 104, 238)],
            "ICSEMANTIC14": [(230, 230, 250)],
            "ICSEMANTIC15": [(0, 0, 255)],
            "ICSEMANTIC16": [(25, 25, 112)],
            "ICSEMANTIC17": [(65, 105, 225)],
            "ICSEMANTIC18": [(176, 196, 222)],
            "ICSEMANTIC19": [(119, 136, 153)],
            "ICSEMANTIC20": [(155, 42, 42)],
            "ICSEMANTIC21": [(200, 0, 0)],
            "ICSEMANTIC22": [(30, 144, 255)],
            "ICSEMANTIC23": [(70, 130, 180)],
            "ICSEMANTIC24": [(0, 191, 255)],
            "ICSEMANTIC25": [(95, 158, 160)],
            "ICSEMANTIC26": [(175, 238, 238)],
            "ICSEMANTIC27": [(0, 255, 255)],
            "ICSEMANTIC28": [(0, 206, 209)],
            "ICSEMANTIC29": [(0, 128, 128)],
            "ICSEMANTIC30": [(127, 255, 170)],
            "ICSEMANTIC31": [(0, 250, 154)],
            "ICSEMANTIC32": [(0, 255, 127)],
            "ICSEMANTIC33": [(60, 179, 113)],
            "ICSEMANTIC34": [(143, 188, 143)],
            "ICSEMANTIC35": [(0, 100, 0)],
            "ICSEMANTIC36": [(124, 252, 0)],
            "ICSEMANTIC37": [(173, 255, 47)],
            "ICSEMANTIC38": [(85, 107, 47)],
            "ICSEMANTIC39": [(245, 245, 220)],
            "ICSEMANTIC40": [(190, 210, 200)],
            "ICSEMANTIC41": [(255, 255, 0)],
            "ICSEMANTIC42": [(189, 183, 107)],
            "ICSEMANTIC43": [(255, 250, 205)],
            "ICSEMANTIC44": [(170, 150, 18)],
            "ICSEMANTIC45": [(240, 230, 140)],
            "ICSEMANTIC46": [(255, 215, 0)],
            "ICSEMANTIC47": [(255, 228, 181)],
            "ICSEMANTIC48": [(255, 165, 0)],
            "ICSEMANTIC49": [(222, 184, 135)],
            "ICSEMANTIC50": [(205, 133, 63)],
            "ICSEMANTIC51": [(255, 218, 185)],
            "ICSEMANTIC52": [(139, 69, 19)],
            "ICSEMANTIC53": [(255, 160, 122)],
            "ICSEMANTIC54": [(255, 69, 0)],
            "ICSEMANTIC55": [(255, 228, 225)],
            "ICSEMANTIC56": [(250, 128, 114)],
        }
    )

    MAPPER_TYPE = list(CODE2COLOR.keys())
    COLOR_TYPE = "RGB"

    __slots__ = ['project_type', 'code2alias', 'code2color', 'merged_idx_mapper']

    def __init__(self,
                 project_type: str,
                 custom_palette: Optional[Dict] = None,
                 update_palette: Optional[Dict] = None,
                 merge_palette: Optional[Dict] = None,
                 **kwargs):

        if project_type not in self.MAPPER_TYPE:
            raise NotImplementedError(f'project_type must be in {self.MAPPER_TYPE}')

        assert len(self.CODE2COLOR[project_type]) == len(self.CODE2ALIAS[project_type]), \
            f'CODE2COLOR({len(self.CODE2COLOR[project_type])}) and ' \
            f'CODE2ALIAS({len(self.CODE2ALIAS[project_type])}) must have same length, '

        self.project_type = project_type
        self.code2alias = None
        self.code2color = None
        self.merged_idx_mapper = None
        self.update_mapper(custom_palette, update_palette, **kwargs)
        self.get_merged_idx_mapper(merge_palette)

    def get_merged_idx_mapper(self, merge_palette: Optional[Dict]) -> NoReturn:
        self.merged_idx_mapper = {i: i for i in range(len(self.code2alias))}
        if merge_palette is not None:
            for key, to_be_merged in merge_palette.items():
                target_key = self.name2idx(key)
                for alias in to_be_merged:
                    ori_idx = self.name2idx(alias)
                    self.merged_idx_mapper[ori_idx] = target_key

    def update_mapper(self,
                      custom_palette: Optional[dict] = None,
                      update_palette: Optional[dict] = None,
                      merge_palette: Optional[dict] = None,
                      **kwargs) -> 'ClassMapper':

        assert not (custom_palette and update_palette), \
            'custom_palette and update_palette cannot be set at the same time'
        if custom_palette is not None:
            self.code2alias = {f'{self.project_type}{i:02d}': [key] for i, key in enumerate(custom_palette.keys())}
            self.code2color = {f'{self.project_type}{i:02d}': value for i, value in enumerate(custom_palette.values())}
        else:
            self.code2alias = deepcopy(self.CODE2ALIAS[self.project_type])
            self.code2color = deepcopy(self.CODE2COLOR[self.project_type])
            if update_palette is not None:
                for code, alias in self.code2alias.items():
                    intersection = set(alias) & set(update_palette.keys())
                    if intersection:
                        if len(intersection) > 1:
                            raise ValueError(f'alias {alias} has more than one intersection with update_palette')
                        self.code2color[code] = update_palette[list(intersection)[0]]
        self.get_merged_idx_mapper(merge_palette)
        return self
    
    def code2idx(self, code: str) -> int:
        return self.merged_idx_mapper[int(code[-2:])]
    
    def name2idx(self, name) -> int:
        return self.code2idx(self.name2code(name))
    
    def name2code(self, name) -> str:
        for code, alias in self.code2alias.items():
            if name in alias:
                return code
        raise ValueError('name not in class mapper')

    def name2color(self, name: str) -> Tuple[int, int, int]:
        for code, alias in self.code2alias.items():
            if name in alias:
                return self.code2color[code][0]
        raise ValueError('name not in class mapper')

    def idx2name(self, idx: int) -> str:
        return self.code2alias[f'{self.project_type}{idx:02d}'][0]

    def idx2color(self, idx: int) -> Tuple[int, int, int]:
        return self.code2color[f'{self.project_type}{idx:02d}'][0]

    def color2idx(self, color: Tuple) -> int:
        for code, colors in self.code2color.items():
            if color in colors:
                return self.code2idx(code)
        raise ValueError(f'color({color}) not in class mapper')

    @property
    def classes(self) -> List[str]:
        return [self.idx2name(k) for k, v in self.merged_idx_mapper.items() if k == v]
    
    @property
    def palettes(self) -> List[Tuple[int, int, int]]:
        return [self.idx2color(k) for k, v in self.merged_idx_mapper.items() if k == v]

    @property
    def num_classes(self) -> int:
        return len(self.code2alias)

    @property
    def rgb_to_idx(self) -> Dict[Tuple[int, int, int], int]:
        return {color: self.code2idx(code) for code, colors in self.code2color.items() for color in colors}

    @property
    def bgr_to_idx(self) -> Dict[Tuple[int, int, int], int]:
        return {color[::-1]: self.code2idx(code) for code, colors in self.code2color.items() for color in colors}

    @property
    def alias_to_idx(self) -> Dict[str, int]:
        return {name: self.code2idx(code) for code, alias in self.code2alias.items() for name in alias}

    @property
    def name_to_idx(self) -> Dict[str, int]:
        return {name: self.code2idx(self.name2code(name)) for name in self.classes}

    @property
    def allowed_colors(self) -> List[Tuple[int, int, int]]:
        return [color for colors in self.code2color.values() for color in colors]


AVICompMapper = ClassMapper('AVICOMP')
AOICompMapper = ClassMapper('AOICOMP')
DefectSegMapper = ClassMapper('DEFSEG')
ICSemanticMapper = ClassMapper('ICSEMANTIC')
AVICompDataMapper = ClassMapper('AVICOMPData')
AVISemanticMapper = ClassMapper('AVISEMANTIC')


if __name__ == '__main__':
    print(len(AVICompMapper.classes))
    AVICompMapper.update_mapper(merge_palette={'Background': ['ViaCircle', 'SurfaceFinish', 'Circuit', 'OpticalPoint']})
    print(len(AVICompMapper.classes))
