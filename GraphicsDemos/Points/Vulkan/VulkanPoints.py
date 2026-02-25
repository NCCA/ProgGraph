#!/usr/bin/env -S uv run --script
import argparse
import os
import sys

import numpy as np
import vulkan as vk
from ncca.ngl import Mat4, Vec3, look_at, perspective
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from NumpyBufferWidget import NumpyBufferWidget


def find_memory_type(
    phys_dev: "vk.PhysicalDevice", type_filter: int, properties: int
) -> int:
    """Finds a suitable memory type for a given filter and properties.

    Args:
        phys_dev: The physical device.
        type_filter: The memory type filter.
        properties: The required memory properties.

    Returns:
        The index of the suitable memory type.

    Raises:
        RuntimeError: If no suitable memory type is found.
    """
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(phys_dev)
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and (
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties
        ):
            return i
    raise RuntimeError("Failed to find suitable memory type")


class VulkanPoints(NumpyBufferWidget):
    """
    A concrete implementation of AbstractWebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self, num_points: int = 10000) -> None:
        """
        Initializes the VulkanPoints class.

        This constructor sets up the Vulkan rendering environment, including the instance,
        physical and logical devices, swapchain, render pass, and graphics pipeline.
        It also creates vertex buffers and command buffers for rendering points.
        """
        super().__init__()
        self.width = 1024
        self.height = 1024
        self.num_points = num_points
        self.setWindowTitle("Vulkan Points")
        self.instance = self._create_instance("Vulkan Points")
        self.view = look_at(Vec3(0, 6, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        gl_to_vulkan = Mat4.from_list(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.projection = gl_to_vulkan @ perspective(
            45.0, self.width / self.height, 0.1, 100.0
        )
        self.rotation = 0.0

        self._select_physical_device()
        self._create_logical_device()
        self._create_offscreen_image()
        self._create_render_pass()
        self._create_framebuffer()
        self._create_shader_modules()
        self._create_pipeline()
        self._create_vertex_buffer()
        self._create_uniform_buffer()
        self._create_command_buffer()
        self.startTimer(16)  # for 60 FPS

    def _create_instance(self, app_name):
        print("Vulkan Environment variables:")
        for var in [
            "VK_ICD_FILENAMES",
            "VK_LAYER_PATH",
            "DYLD_LIBRARY_PATH",
            "VULKAN_SDK",
        ]:
            print(f"  {var}: {os.environ.get(var)}")

        app_info = vk.VkApplicationInfo(
            pApplicationName=app_name,
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="NoEngine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        )

        # Diagnostic: enumerate available extensions
        available_exts = vk.vkEnumerateInstanceExtensionProperties(None)
        print("Available instance extensions:")
        for ext in available_exts:
            print(ext.extensionName)

        # Build a list of required extensions
        required_extensions = ["VK_KHR_surface"]
        if sys.platform == "darwin":
            required_extensions.extend(["VK_MVK_macos_surface", "VK_EXT_metal_surface"])
        print(required_extensions)

        extensions = required_extensions

        flags = 0
        if sys.platform == "darwin":
            extensions.append("VK_KHR_portability_enumeration")
            flags |= vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR

        print("Using extensions:", extensions)

        inst_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions if extensions else None,
            flags=flags,
        )
        return vk.vkCreateInstance(inst_info, None)

    def _select_physical_device(self) -> None:
        """Selects a physical device and graphics queue family."""
        phys_devs = vk.vkEnumeratePhysicalDevices(self.instance)
        self.phys_dev = phys_devs[0]
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.phys_dev)
        self.graphics_queue_index = next(
            i
            for i, q in enumerate(queue_families)
            if q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
        )

    def _create_logical_device(self) -> None:
        """Creates a logical device and gets the graphics queue."""
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.graphics_queue_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        dev_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
        self.device = vk.vkCreateDevice(self.phys_dev, dev_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.graphics_queue_index, 0)

    def _create_offscreen_image(self) -> None:
        """Creates an offscreen image, allocates memory, and creates an image view."""
        img_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_B8G8R8A8_UNORM,
            extent=vk.VkExtent3D(self.width, self.height, 1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self.image = vk.vkCreateImage(self.device, img_info, None)
        mem_reqs = vk.vkGetImageMemoryRequirements(self.device, self.image)
        mem_type_index = find_memory_type(
            self.phys_dev,
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.image_memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindImageMemory(self.device, self.image, self.image_memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self.image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=vk.VK_FORMAT_B8G8R8A8_UNORM,
            components=vk.VkComponentMapping(),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        self.image_view = vk.vkCreateImageView(self.device, view_info, None)

    def _create_render_pass(self) -> None:
        """Creates a render pass."""
        color_attachment = vk.VkAttachmentDescription(
            format=vk.VK_FORMAT_B8G8R8A8_UNORM,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        )
        color_attachment_ref = vk.VkAttachmentReference(
            attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[color_attachment_ref],
        )
        render_pass_info = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=[color_attachment],
            subpassCount=1,
            pSubpasses=[subpass],
        )
        self.render_pass = vk.vkCreateRenderPass(self.device, render_pass_info, None)

    def _create_framebuffer(self) -> None:
        """Creates a framebuffer."""
        fb_info = vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=self.render_pass,
            attachmentCount=1,
            pAttachments=[self.image_view],
            width=self.width,
            height=self.height,
            layers=1,
        )
        self.framebuffer = vk.vkCreateFramebuffer(self.device, fb_info, None)

    def _create_shader_modules(self) -> None:
        """Creates shader modules from SPIR-V files."""

        def load_spirv(path: str) -> bytes:
            """Loads a SPIR-V shader from a file.
            Args:
                path: The path to the SPIR-V file.

            Returns:
                The SPIR-V code as bytes.
            """
            with open(path, "rb") as f:
                return f.read()

        vert_spv = load_spirv("PointVertex.spv")
        frag_spv = load_spirv("PointFragment.spv")

        vert_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(vert_spv),
            pCode=vert_spv,
        )
        self.vert_module = vk.vkCreateShaderModule(self.device, vert_module_info, None)

        frag_module_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(frag_spv),
            pCode=frag_spv,
        )
        self.frag_module = vk.vkCreateShaderModule(self.device, frag_module_info, None)

    def _create_pipeline(self) -> None:
        """Creates the graphics pipeline."""

        # Create a descriptor set layout for the uniform buffer

        ubo_layout_binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
        )

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=1,
            pBindings=[ubo_layout_binding],
        )

        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            self.device, layout_info, None
        )

        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )

        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, pipeline_layout_info, None
        )

        shader_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=self.vert_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=self.frag_module,
                pName="main",
            ),
        ]

        binding_desc = vk.VkVertexInputBindingDescription(
            binding=0,
            stride=6 * 4,  # 6 floats (3 pos, 3 color) * 4 bytes per float
            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
        )

        attr_descs = [
            vk.VkVertexInputAttributeDescription(
                binding=0,
                location=0,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                offset=0,
            ),
            vk.VkVertexInputAttributeDescription(
                binding=0,
                location=1,
                format=vk.VK_FORMAT_R32G32B32_SFLOAT,
                offset=3 * 4,  # Offset by 3 floats (position)
            ),
        ]

        vertex_input_info = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[binding_desc],
            vertexAttributeDescriptionCount=2,
            pVertexAttributeDescriptions=attr_descs,
        )

        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        )

        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            scissorCount=1,
        )

        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_NONE,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
        )

        multisampling = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
        )

        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
            | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT
            | vk.VK_COLOR_COMPONENT_A_BIT
        )

        color_blending = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            attachmentCount=1,
            pAttachments=[color_blend_attachment],
        )

        dynamic_states = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]

        dynamic_state_info = vk.VkPipelineDynamicStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=len(dynamic_states),
            pDynamicStates=dynamic_states,
        )

        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(shader_stages),
            pStages=shader_stages,
            pVertexInputState=vertex_input_info,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=color_blending,
            pDynamicState=dynamic_state_info,
            layout=self.pipeline_layout,
            renderPass=self.render_pass,
            subpass=0,
        )

        self.pipeline = vk.vkCreateGraphicsPipelines(
            self.device, None, 1, [pipeline_info], None
        )[0]

    def _create_vertex_buffer(self) -> None:
        """Creates a vertex buffer and copies vertex data to it."""
        # points = np.random.uniform(-4.0, 4.0, (self.num_points, 3)).astype(np.float32)
        # colours = np.random.uniform(0.0, 1.0, (self.num_points, 3)).astype(np.float32)
        # vertex_data = np.concatenate((points, colours), axis=1)
        vertex_data = np.empty((self.num_points, 6), dtype=np.float32)

        # Populate the first 3 columns with position data
        vertex_data[:, 0:3] = np.random.uniform(-4.0, 4.0, size=(self.num_points, 3))

        # Populate the next 3 columns with colour data
        vertex_data[:, 3:6] = np.random.uniform(0.0, 1.0, size=(self.num_points, 3))

        buffer_size = vertex_data.nbytes
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=buffer_size,
            usage=vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.vertex_buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, self.vertex_buffer)
        mem_type_index = find_memory_type(
            self.phys_dev,
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.vertex_buffer_memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(
            self.device, self.vertex_buffer, self.vertex_buffer_memory, 0
        )
        data_ptr = vk.vkMapMemory(
            self.device, self.vertex_buffer_memory, 0, buffer_size, 0
        )
        dest_ptr = np.frombuffer(data_ptr, dtype=np.float32, count=vertex_data.size)
        np.copyto(dest_ptr, vertex_data.flatten())

        vk.vkUnmapMemory(self.device, self.vertex_buffer_memory)

    def _create_uniform_buffer(self) -> None:
        """Creates a uniform buffer, descriptor pool, and descriptor set."""
        # Create uniform buffer
        buffer_size = 16 * 4  # 4x4 matrix of 4-byte floats
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=buffer_size,
            usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.uniform_buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, self.uniform_buffer)
        mem_type_index = find_memory_type(
            self.phys_dev,
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.uniform_buffer_memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(
            self.device, self.uniform_buffer, self.uniform_buffer_memory, 0
        )

        # Create descriptor pool
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
            maxSets=1,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

        # Allocate descriptor set
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        self.descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

        # Write to descriptor set
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=self.uniform_buffer, offset=0, range=buffer_size
        )
        write_set = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.descriptor_set,
            dstBinding=0,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=[buffer_info],
        )
        vk.vkUpdateDescriptorSets(self.device, 1, [write_set], 0, None)

    def update_uniform_buffer(self) -> None:
        """Updates the uniform buffer with the latest MVP matrix."""
        rotation_matrix = Mat4.rotate_y(self.rotation)
        mvp = self.projection @ self.view @ rotation_matrix
        mvp_np = mvp.to_numpy().astype(np.float32)

        data_ptr = vk.vkMapMemory(self.device, self.uniform_buffer_memory, 0, 16 * 4, 0)
        dest_ptr = np.frombuffer(data_ptr, dtype=np.float32, count=16)
        np.copyto(dest_ptr, mvp_np.flatten())
        vk.vkUnmapMemory(self.device, self.uniform_buffer_memory)

    def _create_command_buffer(self) -> None:
        """Creates a command pool and a command buffer."""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.graphics_queue_index,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.command_buffer = vk.vkAllocateCommandBuffers(self.device, alloc_info)[0]

    def record_and_submit_command_buffer(self) -> None:
        """Records commands to the command buffer, submits it, and waits for completion."""
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        vk.vkBeginCommandBuffer(self.command_buffer, begin_info)

        clear_value = vk.VkClearValue(((0.4, 0.4, 0.4, 1.0),))
        rp_begin_info = vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=self.render_pass,
            framebuffer=self.framebuffer,
            renderArea=[[0, 0], [self.width, self.height]],
            clearValueCount=1,
            pClearValues=[clear_value],
        )
        vk.vkCmdBeginRenderPass(
            self.command_buffer, rp_begin_info, vk.VK_SUBPASS_CONTENTS_INLINE
        )

        viewport = vk.VkViewport(
            x=0, y=0, width=self.width, height=self.height, minDepth=0, maxDepth=1
        )
        vk.vkCmdSetViewport(self.command_buffer, 0, 1, [viewport])
        scissor = vk.VkRect2D(offset=[0, 0], extent=[self.width, self.height])
        vk.vkCmdSetScissor(self.command_buffer, 0, 1, [scissor])

        self.update_uniform_buffer()

        vk.vkCmdBindPipeline(
            self.command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline
        )

        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor_set],
            0,
            None,
        )

        vk.vkCmdBindVertexBuffers(self.command_buffer, 0, 1, [self.vertex_buffer], [0])
        vk.vkCmdDraw(self.command_buffer, self.num_points, 1, 0, 0)
        vk.vkCmdEndRenderPass(self.command_buffer)
        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
            srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
        )
        vk.vkCmdPipelineBarrier(
            self.command_buffer,
            vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            [barrier],
        )

        image_size = self.width * self.height * 4
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=image_size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.staging_buffer = vk.vkCreateBuffer(self.device, buffer_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, self.staging_buffer)
        mem_type_index = find_memory_type(
            self.phys_dev,
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.staging_buffer_memory = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(
            self.device, self.staging_buffer, self.staging_buffer_memory, 0
        )

        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=[0, 0, 0],
            imageExtent=[self.width, self.height, 1],
        )
        vk.vkCmdCopyImageToBuffer(
            self.command_buffer,
            self.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            self.staging_buffer,
            1,
            [region],
        )

        vk.vkEndCommandBuffer(self.command_buffer)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
        )
        fence_info = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = vk.vkCreateFence(self.device, fence_info, None)
        vk.vkQueueSubmit(self.queue, 1, [submit_info], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 1000000000)
        vk.vkDestroyFence(self.device, fence, None)

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        self.buffer = np.zeros([self.height, self.width, 4], dtype=np.uint8)

    def paint(self) -> None:
        """
        Paint the buffer.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(
            10,
            20,
            f"Vulkan Static Points :- {self.num_points}",
            size=20,
            colour=Qt.yellow,
        )
        try:
            self.record_and_submit_command_buffer()

            image_size = self.width * self.height * 4

            buffer = vk.vkMapMemory(
                self.device, self.staging_buffer_memory, 0, image_size, 0
            )

            self.buffer = np.frombuffer(buffer, dtype=np.uint8).reshape(
                (self.height, self.width, 4)
            )

            vk.vkUnmapMemory(self.device, self.staging_buffer_memory)
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def timerEvent(self, event) -> None:
        """
        Handle timer events to update the scene.
        """
        self.rotation += 0.5
        self.update()

    def keyPressEvent(self, event) -> None:
        """
        Handle key press events to control the scene.
        """
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Vulkan points demo")
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=10000,
        help="The number of points to generate.",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = VulkanPoints(num_points=args.points)
    win.show()
    sys.exit(app.exec())
