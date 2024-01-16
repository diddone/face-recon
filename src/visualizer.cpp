#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "shader_class.h"
#include "bfm_manager.h"
#include "utils.h"
#include "face_mesh.h"

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

int main(int argc, char*argv[])
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Help", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    Eigen::Matrix4f transformationMatrix = Eigen::Matrix4f::Zero();
    float scale_factor = 0.5;
    transformationMatrix.diagonal()[0] = scale_factor;
    transformationMatrix.diagonal()[1] = scale_factor;
    transformationMatrix.diagonal()[2] = scale_factor;
    transformationMatrix.diagonal()[3] = 1.0f;

    float left = -250, right = 250, top = 250, bottom = -250;
    float zFar = 1000;
    float zNear = 0.01;

    Eigen::Matrix4f projectionMatrix;
    projectionMatrix <<
        2/(right - left), 0, 0, 0,
        0, 2/(top - bottom), 0, - (top + bottom) / (top - bottom),
        0, 0, -2/(zFar - zNear), -(zFar + zNear)/(zFar - zNear),
        0, 0, 0, 1;

    transformationMatrix = projectionMatrix * transformationMatrix;

    // build and compile our shader program
    // ------------------------------------
    Shader imageShader("/home/david/Documents/tum/projects/3d/face-recon/src/shaders/4.1.texture.vs", "/home/david/Documents/tum/projects/3d/face-recon/src/shaders/4.1.texture.fs");
    Shader faceMeshShader("/home/david/Documents/tum/projects/3d/face-recon/src/shaders/4.1.texture.mesh.vs", "/home/david/Documents/tum/projects/3d/face-recon/src/shaders/4.1.texture.mesh.fs");
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;

    stbi_set_flip_vertically_on_load(true);

    unsigned char *data = stbi_load("/home/david/Documents/tum/projects/3d/face-recon/src/image.jpeg", &width, &height, &nrChannels, 3);

    float aspect_ratio = float(height) / float(width);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float imageFrameVertices[] = {
             // positions                 // colors           // texture coords
             1.0f,  aspect_ratio, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
             1.0f, -aspect_ratio, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
            -1.0f, -aspect_ratio, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
            -1.0f,  aspect_ratio, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
    };
    unsigned int imageFrameIndices[] = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(imageFrameVertices), imageFrameVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(imageFrameIndices), imageFrameIndices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // load and create a texture
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // face stuff

    std::string sBfmH5Path = "../Data/model2017-1_face12_nomouth.h5",
            sLandmarkIdxPath = "../example/example_landmark_68.anl";
    double dFx = 1744.327628674942, dFy = 1747.838275588676, dCx = 800., dCy = 600.;
    bool isInitGlog = initGlog(argc, argv);
    if (!isInitGlog) {
        std::cout << "Glog problem or just info help\n";
        return 1;
    }
    std::unique_ptr<BfmManager> pBfmManager(new BfmManager(sBfmH5Path, sLandmarkIdxPath));
    std::cout << "H% file path " << sBfmH5Path << "\n";
    std::cout << "faces\n\n";
    pBfmManager->genAvgFace();
    pBfmManager->writePly("avg_face.ply");

//    pBfmManager->genRndFace(1.0);
//    pBfmManager->writePly("rnd_face.ply");
    std::cout << "Number of imageFrameVertices " << pBfmManager->m_nVertices << "\n";
    std::cout << "Number of faces " << pBfmManager->m_nFaces << "\n";

    std::vector<Vertex> face_vertices;
    std::vector<unsigned int> face_indices;

    for (size_t iVertex = 0; iVertex < pBfmManager->m_nVertices; iVertex++) {
        float x, y, z, r, g, b;

        x = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3));
        y = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3 + 1));
        z = float(pBfmManager->m_vecCurrentBlendshape(iVertex * 3 + 2));

        r = float(pBfmManager->m_vecCurrentTex(iVertex * 3));
        g = float(pBfmManager->m_vecCurrentTex(iVertex * 3 + 1));
        b = float(pBfmManager->m_vecCurrentTex(iVertex * 3 + 2));

        face_vertices.push_back({x, y, z, r, g, b });
    }

    for (size_t iIndex = 0; iIndex < pBfmManager->m_nFaces; iIndex++) {
        face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3));
        face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3 + 1));
        face_indices.push_back(pBfmManager->m_vecTriangleList(iIndex * 3 + 2));
    }

    Mesh face_mesh(face_vertices, face_indices);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // bind Texture
        glBindTexture(GL_TEXTURE_2D, texture);

        // render container
        imageShader.use();

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        faceMeshShader.use();
        faceMeshShader.setMatrix("transformation", transformationMatrix);

        face_mesh.Draw(faceMeshShader);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}