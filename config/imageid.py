evaluator = dict(
    type='ImageidEvaluator',
    metrics=dict(
        fid=dict(
            type='FIDScore',
            dims=2048,
            batch_size=32,
            num_workers=1,
        ),
        aesthetic=dict(
            type='AestheticScore',
            batch_size=4,
        ),
        imaging_quality=dict(
            type='ImagingQuality',
            batch_size=4,
        ),
        facesim=dict(
            type='FaceSim',
        ),
        antifacesim=dict(
            type='AntiFaceSim',
        ),
        landmarkdiff=dict(
            type='LandmarkDiff',
        ),
        # facediv1 = dict(
        #     type='FaceDiv1',
        # ),
        # facediv2 = dict(
        #     type='FaceDiv2',
        # ),
        clipi=dict(
            type='ClipI',
        ),
        clipt=dict(
            type='ClipT',
            truncate=True
        ),
        # dino=dict(
        #     type='DINO',
        # ),
        # fgis = dict(
        #     type='FGIS',
        # ),
        # dreamsim=dict(
        #     type='DreamSim',
        # ),
        posediv=dict(
            type='PoseDiv',
        ),
        exprdiv=dict(
            type='ExprDiv',
        ),
        # gpt=dict(
        #     type='GPT',
        #     model="gpt-4O-mini",
        #     user_prompt="",
        # )

    ),
    visual=dict(
        visualposediv=dict(
            type='VisualPoseDiv',
            # tdx=200,
            # tdy=200,
            size=100
        )
    )
)
